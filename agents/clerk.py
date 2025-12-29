"""
Clerk Agent - Location Services

Role: Find nearby legal resources (police stations, courts, legal aid)
Goal: Help users locate physical legal services in their area
Tools: Google Maps API, OpenStreetMap fallback
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger
import os


@dataclass
class LegalResource:
    """A nearby legal resource"""
    name: str
    address: str
    phone: Optional[str] = None
    hours: Optional[str] = None
    distance_km: float = 0.0
    latitude: float = 0.0
    longitude: float = 0.0
    maps_url: Optional[str] = None
    resource_type: str = "police_station"  # court, legal_aid, advocate


class ClerkAgent:
    """
    The Clerk Agent handles location-based services.
    
    Workflow:
    1. Geocode user's location (if address provided)
    2. Search for nearby legal resources
    3. Filter and sort by distance
    4. Return top 5 with contact details
    """
    
    RESOURCE_TYPES = {
        "police": "police station",
        "court": "district court OR high court OR magistrate court",
        "legal_aid": "legal aid center OR legal services OR free legal",
        "advocate": "lawyer OR advocate OR attorney",
    }
    
    def __init__(
        self,
        google_maps_api_key: Optional[str] = None,
        search_radius_km: int = 5,
        max_results: int = 5
    ):
        self.api_key = google_maps_api_key or os.getenv("GOOGLE_MAPS_API_KEY")
        self.radius = search_radius_km
        self.max_results = max_results
        
        # Initialize Google Maps client if key available
        self.gmaps = None
        if self.api_key:
            try:
                import googlemaps
                self.gmaps = googlemaps.Client(key=self.api_key)
                logger.info("Google Maps client initialized")
            except ImportError:
                logger.warning("googlemaps package not installed")
    
    async def find_resources(
        self,
        latitude: float,
        longitude: float,
        resource_type: str = "police"
    ) -> List[LegalResource]:
        """
        Find nearby legal resources.
        
        Args:
            latitude: User's latitude
            longitude: User's longitude
            resource_type: Type of resource (police, court, legal_aid, advocate)
            
        Returns:
            List of nearby resources sorted by distance
        """
        logger.info(f"Clerk searching for {resource_type} near ({latitude}, {longitude})")
        
        if not self.gmaps:
            logger.warning("Google Maps not configured, returning empty results")
            return []
        
        search_query = self.RESOURCE_TYPES.get(resource_type, resource_type)
        
        # TODO: Implement actual Google Places search
        # results = self.gmaps.places_nearby(
        #     location=(latitude, longitude),
        #     radius=self.radius * 1000,  # Convert km to meters
        #     keyword=search_query
        # )
        
        results = []
        logger.info(f"Found {len(results)} {resource_type} resources")
        return results[:self.max_results]
    
    async def geocode_address(self, address: str) -> Optional[Dict[str, float]]:
        """Convert address to coordinates"""
        if not self.gmaps:
            return None
        
        # TODO: Implement geocoding
        # result = self.gmaps.geocode(address)
        # if result:
        #     location = result[0]["geometry"]["location"]
        #     return {"lat": location["lat"], "lng": location["lng"]}
        
        return None
    
    def _calculate_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """Calculate distance between two coordinates in km (Haversine)"""
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371  # Earth's radius in km
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c


def create_clerk_agent(**kwargs) -> ClerkAgent:
    """Factory function to create a Clerk agent"""
    return ClerkAgent(**kwargs)
