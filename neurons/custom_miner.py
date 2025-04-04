import time
import torch
import typing
import bittensor as bt
import openmeteo_requests
import numpy as np
import pandas as pd
import requests_cache
from retry_requests import retry

from zeus.utils.misc import celcius_to_kelvin, is_updated
from zeus.utils.config import get_device_str
from zeus.utils.time import get_timestamp
from zeus.protocol import TimePredictionSynapse
from zeus.base.miner import BaseMinerNeuron
from zeus import __version__ as zeus_version


class Miner(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior.
    In particular, you should replace the forward function with your own logic.
    You may also want to override the blacklist and priority functions according to your needs.

    Currently the miner simply does a request to OpenMeteo (https://open-meteo.com/) asking for a prediction.
    You are encouraged to attempt to improve over this by changing the forward function.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        self.device: torch.device = torch.device(get_device_str())
        
        # Setup OpenMeteo API with caching and retry mechanisms
        # Cache API responses to reduce rate limiting issues
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)  # 1 hour cache
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.openmeteo_api = openmeteo_requests.Client(session=retry_session)

    async def forward(self, synapse: TimePredictionSynapse) -> TimePredictionSynapse:
        """
        Processes the incoming TimePredictionSynapse using OpenMeteo's enhanced API.
        """
        # Get coordinates from synapse
        coordinates = torch.Tensor(synapse.locations)
        start_time = get_timestamp(synapse.start_time)
        end_time = get_timestamp(synapse.end_time)
        
        bt.logging.info(f"Received request from {synapse.dendrite.hotkey[:5]}")
        bt.logging.info(f"Predicting {synapse.requested_hours} hours for grid of shape {coordinates.shape}.")
        
        # Extract latitude and longitude
        latitudes, longitudes = coordinates.view(-1, 2).T
        
        # Set up enhanced API parameters
        params = {
            "latitude": latitudes.tolist(),
            "longitude": longitudes.tolist(),
            "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", 
                      "cloud_cover", "wind_speed_10m", "soil_temperature_0cm",
                      "surface_pressure"],
            "start_date": start_time.strftime("%Y-%m-%d"),
            "end_date": end_time.strftime("%Y-%m-%d"),
            #"elevation": "nan",  # Use grid cell average elevation
            "cell_selection": "land",  # Prefer land cells with similar elevation
            "models": "best_match",  # Let API choose best models
            "timezone": "GMT",  # Ensure consistent timezone
            "apikey": "api_key_here"
        }
        
        try:
            # Make API request
            responses = self.openmeteo_api.weather_api(
                "https://customer-api.open-meteo.com/v1/forecast", params=params ## this URL is only for paid subs - free is: api.open-meteo.com/v1/forecast
            )
            
            # Extract all variables from responses
            var_count = len(params["hourly"])
            all_data = []
            
            for var_idx in range(var_count):
                var_data = np.stack(
                    [r.Hourly().Variables(var_idx).ValuesAsNumpy() for r in responses], axis=1
                ).reshape(-1, coordinates.shape[0], coordinates.shape[1])
                all_data.append(var_data)
            
            # Base prediction from temperature_2m
            temp_data = all_data[0]  # temperature_2m is first in our list
            output = celcius_to_kelvin(temp_data)
            
            # Extract the hours we need
            for i in range(len(all_data)):
                all_data[i] = all_data[i][start_time.hour:start_time.hour + synapse.requested_hours]
            output = output[start_time.hour:start_time.hour + synapse.requested_hours]
            
            # Apply corrections based on other variables
            humidity_data = all_data[1]  # relative_humidity_2m
            dew_point_data = all_data[2]  # dew_point_2m
            cloud_data = all_data[3]  # cloud_cover
            wind_data = all_data[4]  # wind_speed_10m
            soil_temp_data = all_data[5]  # soil_temperature_0cm
            
            output = self._apply_weather_corrections(
                output, humidity_data, dew_point_data, cloud_data, wind_data, soil_temp_data
            )
            
        except Exception as e:
            bt.logging.warning(f"OpenMeteo API error: {e}")
            # Fallback to simple climatology
            output = self._get_climatology_fallback(coordinates, start_time, synapse.requested_hours)
        
        # Apply smoothing
        output = self._apply_smoothing(output)
        
        bt.logging.info(f"Output shape is {output.shape}")
        synapse.predictions = output.tolist()
        synapse.version = zeus_version
        
        return synapse

    def _apply_weather_corrections(self, temp, humidity, dew_point, cloud, wind, soil_temp):
        """Apply physics-informed corrections based on other weather variables"""
        corrected_temp = temp.copy()
        
        # 1. Cloud cover correction - improved with research-backed values
        for i in range(temp.shape[0]):
            current_hour = (i % 24)
            is_night = (current_hour < 6) or (current_hour > 18)
            
            if is_night:
                # Convert humidity to dryness factor (drier air = more cooling)
                rel_humidity = humidity[i] / 100.0
                dryness_factor = 1.0 - (0.7 * rel_humidity)  # Ranges from 0.3-1.0
                
                # Research shows clear nights can be 2-5°C cooler than cloudy nights
                cloud_factor = cloud[i] / 100.0
                # Use 3°C as average maximum effect, scaled by dryness
                night_cooling = -3.0 * (1.0 - cloud_factor) * dryness_factor
                corrected_temp[i] += night_cooling
        
        # 2. Wind speed effect - logarithmic relationship from boundary layer studies
        for i in range(temp.shape[0]):
            # Convert wind using logarithmic scale (as per boundary layer studies)
            wind_speeds = np.maximum(wind[i], 0.1)  # Avoid log(0)
            wind_factor = np.log1p(wind_speeds) / np.log(21.0)  # log(1+x)/log(21) gives 0-1 range
            
            # Calculate temperature deviation from mean
            if i >= 24:
                daily_mean = np.mean(temp[i-24:i], axis=0)
            else:
                daily_mean = np.mean(temp[:i+1], axis=0)
            
            temp_deviation = corrected_temp[i] - daily_mean
            
            # Strong winds reduce deviation by up to 35% (research-backed value)
            dampened_deviation = temp_deviation * (1.0 - 0.35 * wind_factor)
            corrected_temp[i] = daily_mean + dampened_deviation
        
        # 3. Soil temperature influence - varies by time and conditions
        for i in range(temp.shape[0]):
            current_hour = (i % 24)
            
            # Dawn hours have strongest soil influence
            if 3 <= current_hour <= 8:
                # Soil influence varies from 0.2 to 0.45 (based on research)
                # Dawn influence peaks at around 0.4
                dawn_factor = 1.0 - abs((current_hour - 5.5) / 2.5)  # Peaks at 5:30am
                soil_influence = 0.2 + (0.2 * dawn_factor)
                
                soil_temp_kelvin = celcius_to_kelvin(soil_temp[i])
                soil_diff = soil_temp_kelvin - corrected_temp[i]
                corrected_temp[i] += soil_influence * soil_diff
        
        # 4. Humidity effect - non-linear impact on temperature changes
        for i in range(1, temp.shape[0]):
            # Non-linear relationship with stronger effect at high humidity
            humidity_factor = humidity[i] / 100.0
            humidity_effect = 0.15 + 0.25 * (humidity_factor ** 2)
            
            # Calculate temperature change
            raw_change = corrected_temp[i] - corrected_temp[i-1]
            
            # Apply dampening effect
            dampened_change = raw_change * (1.0 - humidity_effect)
            corrected_temp[i] = corrected_temp[i-1] + dampened_change
        
        return corrected_temp

    def _apply_smoothing(self, output):
        """Apply time-series smoothing to reduce noise"""
        kernel_size = 3
        smooth_output = np.zeros_like(output)
        
        for i in range(output.shape[0]):
            idx_start = max(0, i - kernel_size // 2)
            idx_end = min(output.shape[0], i + kernel_size // 2 + 1)
            smooth_output[i] = np.mean(output[idx_start:idx_end], axis=0)
        
        return smooth_output

    def _get_climatology_fallback(self, coordinates, start_time, requested_hours):
        """Provide fallback predictions if API fails"""
        lat_grid = coordinates[..., 0].numpy()
        
        # Day of year (0-365)
        day_of_year = start_time.timetuple().tm_yday
        
        # Seasonal factor (peak in summer, low in winter for northern hemisphere)
        seasonal_factor = np.cos(2 * np.pi * (day_of_year - 182) / 365)
        
        # Base temperature + latitude effect + seasonal effect
        base_temp = 288.0  # ~15°C in Kelvin
        lat_effect = -0.4 * np.abs(lat_grid)  # Temperature drops toward poles
        seasonal_effect = 15.0 * seasonal_factor * (1.0 - np.abs(lat_grid) / 90.0)
        
        # Combine effects
        temp = base_temp + lat_effect + seasonal_effect
        
        # Create output array
        output = np.zeros((requested_hours, coordinates.shape[0], coordinates.shape[1]))
        for i in range(requested_hours):
            output[i] = temp
        
        # Add simple diurnal cycle - temperature peak around 14:00, low around 02:00
        hours = [(start_time + pd.Timedelta(hours=i)).hour for i in range(requested_hours)]
        diurnal_pattern = np.array([np.sin(np.pi * (h - 2) / 12) for h in hours])
        
        # Apply diurnal pattern
        for i, pattern_value in enumerate(diurnal_pattern):
            # Amplitude varies by latitude (stronger at equator)
            amplitude = 5.0 * (1.0 - np.abs(lat_grid) / 90.0)
            output[i] += pattern_value * amplitude
        
        return output

    async def blacklist(
        self, synapse: TimePredictionSynapse
    ) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contracted via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (template.protocol.Dummy): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return True, "Missing dendrite or hotkey"

        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if (
            not self.config.blacklist.allow_non_registered
            and synapse.dendrite.hotkey not in self.metagraph.hotkeys
        ):
            # Ignore requests from un-registered entities.
            bt.logging.trace(
                f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}"
            )
            return True, "Unrecognized hotkey"

        if self.config.blacklist.force_validator_permit:
            # If the config is set to force validator permit, then we should only allow requests from validators.
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(
                    f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}"
                )
                return True, "Non-validator hotkey"
            
        if self.metagraph.S[uid] < self.config.blacklist.minimal_alpha_stake:
            # require true validators to have at least minimal alpha stake.
            bt.logging.warning(
                f"Blacklisting a request from hotkey {synapse.dendrite.hotkey} with only {self.metagraph.S[uid]} stake."
            )
            return True, "Non-validator hotkey"

        bt.logging.trace(
            f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}"
        )
        return False, "Hotkey recognized!"

    async def priority(self, synapse: TimePredictionSynapse) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (template.protocol.Dummy): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may receive messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return 0.0

        caller_uid = self.metagraph.hotkeys.index(
            synapse.dendrite.hotkey
        )  # Get the caller index.
        priority = float(
            self.metagraph.S[caller_uid]
        )  # Return the stake as the priority.
        bt.logging.trace(
            f"Prioritizing {synapse.dendrite.hotkey} with value: {priority}"
        )
        return priority


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info(f"Miner running | uid {miner.uid} | {time.time()}")
            time.sleep(5)