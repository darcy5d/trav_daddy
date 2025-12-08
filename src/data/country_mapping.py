"""
City to Country mapping for venue hierarchical display.

This module provides mapping from cities to their countries and country emoji flags.
"""

# Country emoji flags (ISO 3166-1 alpha-2 codes)
COUNTRY_FLAGS = {
    "Afghanistan": "🇦🇫",
    "Argentina": "🇦🇷",
    "Australia": "🇦🇺",
    "Austria": "🇦🇹",
    "Bahrain": "🇧🇭",
    "Bangladesh": "🇧🇩",
    "Barbados": "🇧🇧",
    "Belgium": "🇧🇪",
    "Bhutan": "🇧🇹",
    "Botswana": "🇧🇼",
    "Brazil": "🇧🇷",
    "Bulgaria": "🇧🇷",
    "Cambodia": "🇰🇭",
    "Canada": "🇨🇦",
    "Chile": "🇨🇱",
    "China": "🇨🇳",
    "Costa Rica": "🇨🇷",
    "Croatia": "🇭🇷",
    "Cyprus": "🇨🇾",
    "Czech Republic": "🇨🇿",
    "Denmark": "🇩🇰",
    "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "Estonia": "🇪🇪",
    "Eswatini": "🇸🇿",
    "Fiji": "🇫🇯",
    "Finland": "🇫🇮",
    "France": "🇫🇷",
    "Germany": "🇩🇪",
    "Ghana": "🇬🇭",
    "Gibraltar": "🇬🇮",
    "Greece": "🇬🇷",
    "Guernsey": "🇬🇬",
    "Guyana": "🇬🇾",
    "Hong Kong": "🇭🇰",
    "Hungary": "🇭🇺",
    "India": "🇮🇳",
    "Indonesia": "🇮🇩",
    "Ireland": "🇮🇪",
    "Isle of Man": "🇮🇲",
    "Italy": "🇮🇹",
    "Jamaica": "🇯🇲",
    "Japan": "🇯🇵",
    "Jersey": "🇯🇪",
    "Kenya": "🇰🇪",
    "Kuwait": "🇰🇼",
    "Luxembourg": "🇱🇺",
    "Malawi": "🇲🇼",
    "Malaysia": "🇲🇾",
    "Maldives": "🇲🇻",
    "Malta": "🇲🇹",
    "Mexico": "🇲🇽",
    "Mozambique": "🇲🇿",
    "Namibia": "🇳🇦",
    "Nepal": "🇳🇵",
    "Netherlands": "🇳🇱",
    "New Caledonia": "🇳🇨",
    "New Zealand": "🇳🇿",
    "Nigeria": "🇳🇬",
    "Northern Ireland": "🇬🇧",
    "Norway": "🇳🇴",
    "Oman": "🇴🇲",
    "Pakistan": "🇵🇰",
    "Panama": "🇵🇦",
    "Papua New Guinea": "🇵🇬",
    "Philippines": "🇵🇭",
    "Portugal": "🇵🇹",
    "Qatar": "🇶🇦",
    "Romania": "🇷🇴",
    "Rwanda": "🇷🇼",
    "Samoa": "🇼🇸",
    "Saudi Arabia": "🇸🇦",
    "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "Serbia": "🇷🇸",
    "Sierra Leone": "🇸🇱",
    "Singapore": "🇸🇬",
    "South Africa": "🇿🇦",
    "South Korea": "🇰🇷",
    "Spain": "🇪🇸",
    "Sri Lanka": "🇱🇰",
    "St Kitts": "🇰🇳",
    "St Lucia": "🇱🇨",
    "St Vincent": "🇻🇨",
    "Sweden": "🇸🇪",
    "Tanzania": "🇹🇿",
    "Thailand": "🇹🇭",
    "Trinidad": "🇹🇹",
    "UAE": "🇦🇪",
    "Uganda": "🇺🇬",
    "USA": "🇺🇸",
    "Vanuatu": "🇻🇺",
    "Wales": "🏴󠁧󠁢󠁷󠁬󠁳󠁿",
    "West Indies": "🌴",
    "Zambia": "🇿🇲",
    "Zimbabwe": "🇿🇼",
}

# City to country mapping
CITY_TO_COUNTRY = {
    # Afghanistan
    "Kabul": "Afghanistan",
    
    # Argentina
    "Buenos Aires": "Argentina",
    
    # Australia
    "Adelaide": "Australia",
    "Alice Springs": "Australia",
    "Albury": "Australia",
    "Ballarat": "Australia",
    "Brisbane": "Australia",
    "Burnie": "Australia",
    "Cairns": "Australia",
    "Canberra": "Australia",
    "Carrara": "Australia",
    "Coffs Harbour": "Australia",
    "Darwin": "Australia",
    "Geelong": "Australia",
    "Hobart": "Australia",
    "Latrobe": "Australia",
    "Launceston": "Australia",
    "Mackay": "Australia",
    "Melbourne": "Australia",
    "Moe": "Australia",
    "Nuriootpa": "Australia",
    "Perth": "Australia",
    "Sydney": "Australia",
    "Townsville": "Australia",
    
    # Austria
    "Graz": "Austria",
    "Latschach": "Austria",
    "Lower Austria": "Austria",
    
    # Bangladesh
    "Chattogram": "Bangladesh",
    "Dhaka": "Bangladesh",
    "Mirpur": "Bangladesh",
    "Sylhet": "Bangladesh",
    
    # Belgium
    "Ghent": "Belgium",
    "Zemst": "Belgium",
    
    # Botswana
    "Gaborone": "Botswana",
    
    # Brazil
    "Seropedica": "Brazil",
    
    # Bulgaria
    "Sofia": "Bulgaria",
    
    # Cambodia
    "Phnom Penh": "Cambodia",
    
    # Canada
    "King City": "Canada",
    "Milverton": "Canada",
    "Waterloo": "Canada",
    
    # China
    "Hangzhou": "China",
    
    # Costa Rica
    "Guacima": "Costa Rica",
    
    # Croatia
    "Zagreb": "Croatia",
    
    # Cyprus
    "Episkopi": "Cyprus",
    
    # Czech Republic
    "Prague": "Czech Republic",
    
    # Denmark
    "Brondby": "Denmark",
    "Copenhagen": "Denmark",
    "Ishoj": "Denmark",
    "Koge": "Denmark",
    
    # England
    "Arundel": "England",
    "Beckenham": "England",
    "Birmingham": "England",
    "Blackpool": "England",
    "Brighton": "England",
    "Bristol": "England",
    "Cambridge": "England",
    "Canterbury": "England",
    "Chelmsford": "England",
    "Cheltenham": "England",
    "Chester": "England",
    "Chester-le-Street": "England",
    "Chesterfield": "England",
    "Derby": "England",
    "Eastbourne": "England",
    "Gosforth": "England",
    "Grantham": "England",
    "Guildford": "England",
    "Hove": "England",
    "Kibworth": "England",
    "Kidderminster": "England",
    "Leeds": "England",
    "Leicester": "England",
    "Lincoln": "England",
    "Liverpool": "England",
    "London": "England",
    "Loughborough": "England",
    "Manchester": "England",
    "Neath": "England",
    "Nettleworth": "England",
    "Newbury": "England",
    "Newport": "England",
    "Northampton": "England",
    "Northwood": "England",
    "Nottingham": "England",
    "Radlett": "England",
    "Richmond": "England",
    "Rugby": "England",
    "Sale": "England",
    "Scarborough": "England",
    "Sedbergh": "England",
    "Solihull": "England",
    "Sookholme": "England",
    "Southampton": "England",
    "Southport": "England",
    "Street": "England",
    "Taunton": "England",
    "Uxbridge": "England",
    "Welbeck": "England",
    "Worcester": "England",
    "Wormsley": "England",
    "York": "England",
    
    # Estonia
    "Tallinn": "Estonia",
    
    # Eswatini
    "Malkerns": "Eswatini",
    
    # Fiji
    "Suva": "Fiji",
    
    # Finland
    "Kerava": "Finland",
    "Vantaa": "Finland",
    
    # France
    "Dreux": "France",
    
    # Germany
    "Krefeld": "Germany",
    
    # Ghana
    "Accra": "Ghana",
    
    # Gibraltar
    "Gibraltar": "Gibraltar",
    
    # Greece
    "Corfu": "Greece",
    
    # Guernsey
    "Castel": "Guernsey",
    "St Peter Port": "Guernsey",
    
    # Guyana
    "Georgetown": "Guyana",
    "Guyana": "Guyana",
    "Providence": "Guyana",
    
    # Hong Kong
    "Hong Kong": "Hong Kong",
    "Kowloon": "Hong Kong",
    "Mong Kok": "Hong Kong",
    "Wong Nai Chung Gap": "Hong Kong",
    
    # Hungary
    "Szodliget": "Hungary",
    
    # India
    "Ahmedabad": "India",
    "Bengaluru": "India",
    "Chandigarh": "India",
    "Chennai": "India",
    "Cuttack": "India",
    "Dehra Dun": "India",
    "Delhi": "India",
    "Dharamsala": "India",
    "Guwahati": "India",
    "Gwalior": "India",
    "Hyderabad": "India",
    "Indore": "India",
    "Jaipur": "India",
    "Kolkata": "India",
    "Lucknow": "India",
    "Mohali": "India",
    "Mumbai": "India",
    "Nagpur": "India",
    "Navi Mumbai": "India",
    "New Chandigarh": "India",
    "Pune": "India",
    "Raipur": "India",
    "Rajkot": "India",
    "Ranchi": "India",
    "Surat": "India",
    "Thiruvananthapuram": "India",
    "Vadodara": "India",
    "Visakhapatnam": "India",
    
    # Indonesia
    "Bali": "Indonesia",
    
    # Ireland
    "Bready": "Ireland",
    "Cork": "Ireland",
    "Dublin": "Ireland",
    "Wicklow": "Ireland",
    
    # Isle of Man
    # (No cities in database)
    
    # Italy
    "Rome": "Italy",
    "Spinaceto": "Italy",
    
    # Jamaica
    "Jamaica": "Jamaica",
    "Kingston": "Jamaica",
    
    # Japan
    "Osaka": "Japan",
    "Sano": "Japan",
    
    # Jersey
    "St Clement": "Jersey",
    "St Saviour": "Jersey",
    
    # Kenya
    "Nairobi": "Kenya",
    
    # Luxembourg
    "Walferdange": "Luxembourg",
    
    # Malawi
    "Blantyre": "Malawi",
    
    # Malaysia
    "Bangi": "Malaysia",
    "Johor": "Malaysia",
    "Kuala Lumpur": "Malaysia",
    
    # Malta
    "Marsa": "Malta",
    
    # Mexico
    "Mexico City": "Mexico",
    "Naucalpan": "Mexico",
    
    # Namibia
    "Windhoek": "Namibia",
    
    # Nepal
    "Kathmandu": "Nepal",
    "Kirtipur": "Nepal",
    "Pokhara": "Nepal",
    
    # Netherlands
    "Amstelveen": "Netherlands",
    "Deventer": "Netherlands",
    "Rotterdam": "Netherlands",
    "Schiedam": "Netherlands",
    "The Hague": "Netherlands",
    "Utrecht": "Netherlands",
    
    # New Caledonia
    "Noumea": "New Caledonia",
    
    # New Zealand
    "Auckland": "New Zealand",
    "Christchurch": "New Zealand",
    "Dunedin": "New Zealand",
    "Hamilton": "New Zealand",
    "Lincoln": "New Zealand",
    "Mount Maunganui": "New Zealand",
    "Napier": "New Zealand",
    "Nelson": "New Zealand",
    "New Plymouth": "New Zealand",
    "Palmerston North": "New Zealand",
    "Queenstown": "New Zealand",
    "Wellington": "New Zealand",
    "Whangarei": "New Zealand",
    
    # Nigeria
    "Abuja": "Nigeria",
    "Lagos": "Nigeria",
    
    # Northern Ireland
    "Belfast": "Northern Ireland",
    "Comber": "Northern Ireland",
    "Derry": "Northern Ireland",
    "Eglinton": "Northern Ireland",
    
    # Norway
    "Oslo": "Norway",
    
    # Oman
    "Al Amarat": "Oman",
    
    # Pakistan
    "Faisalabad": "Pakistan",
    "Karachi": "Pakistan",
    "Lahore": "Pakistan",
    "Multan": "Pakistan",
    "Rawalpindi": "Pakistan",
    
    # Panama
    "Panama City": "Panama",
    
    # Papua New Guinea
    "Port Moresby": "Papua New Guinea",
    
    # Portugal
    "Albergaria": "Portugal",
    
    # Qatar
    "Doha": "Qatar",
    
    # Romania
    "Ilfov County": "Romania",
    
    # Rwanda
    "Kigali City": "Rwanda",
    
    # Samoa
    "Apia": "Samoa",
    
    # Scotland
    "Aberdeen": "Scotland",
    "Arbroath": "Scotland",
    "Ayr": "Scotland",
    "Dundee": "Scotland",
    "Edinburgh": "Scotland",
    "Glasgow": "Scotland",
    
    # Serbia
    "Belgrade": "Serbia",
    
    # Singapore
    "Singapore": "Singapore",
    
    # South Africa
    "Benoni": "South Africa",
    "Bloemfontein": "South Africa",
    "Cape Town": "South Africa",
    "Centurion": "South Africa",
    "Durban": "South Africa",
    "East London": "South Africa",
    "Gqeberha": "South Africa",
    "Johannesburg": "South Africa",
    "Kimberley": "South Africa",
    "Paarl": "South Africa",
    "Pietermaritzburg": "South Africa",
    "Port Elizabeth": "South Africa",
    "Potchefstroom": "South Africa",
    "Pretoria": "South Africa",
    
    # South Korea
    "Incheon": "South Korea",
    
    # Spain
    "Almeria": "Spain",
    "Murcia": "Spain",
    
    # Sri Lanka
    "Colombo": "Sri Lanka",
    "Dambulla": "Sri Lanka",
    "Galle": "Sri Lanka",
    "Hambantota": "Sri Lanka",
    "Kaluthara": "Sri Lanka",
    "Kandy": "Sri Lanka",
    "Katunayake": "Sri Lanka",
    "Kurunegala": "Sri Lanka",
    "Maggona": "Sri Lanka",
    "Moratuwa": "Sri Lanka",
    "Panadura": "Sri Lanka",
    "Panagoda": "Sri Lanka",
    
    # St Kitts and Nevis
    "Basseterre": "St Kitts",
    "St Kitts": "St Kitts",
    
    # St Lucia
    "Gros Islet": "St Lucia",
    "St Lucia": "St Lucia",
    
    # St Vincent
    "Kingstown": "St Vincent",
    
    # Sweden
    "Kolsva": "Sweden",
    "Stockholm": "Sweden",
    
    # Tanzania
    "Dar-es-Salaam": "Tanzania",
    
    # Thailand
    "Bangkok": "Thailand",
    "Chiang Mai": "Thailand",
    
    # Trinidad and Tobago
    "Port of Spain": "Trinidad",
    "Tarouba": "Trinidad",
    "Trinidad": "Trinidad",
    
    # UAE
    "Abu Dhabi": "UAE",
    "Ajman": "UAE",
    "Dubai": "UAE",
    "Sharjah": "UAE",
    
    # Uganda
    "Entebbe": "Uganda",
    "Jinja": "Uganda",
    "Kampala": "Uganda",
    
    # USA
    "Dallas": "USA",
    "Grand Prairie": "USA",
    "Houston": "USA",
    "Lauderhill": "USA",
    "Los Angeles": "USA",
    "Morrisville": "USA",
    "New York": "USA",
    "Oakland": "USA",
    "Pearland": "USA",
    
    # Vanuatu
    "Port Vila": "Vanuatu",
    
    # Wales
    "Cardiff": "Wales",
    
    # West Indies (generic/multi-island)
    "Antigua": "West Indies",
    "Barbados": "West Indies",
    "Cave Hill": "West Indies",
    "Coolidge": "West Indies",
    "George Town": "West Indies",
    "North Sound": "West Indies",
    "Roseau": "West Indies",
    "St George's": "West Indies",
    "St Martin": "West Indies",
    
    # Zimbabwe
    "Bulawayo": "Zimbabwe",
    "Harare": "Zimbabwe",
    "Alexandra": "Zimbabwe",
}


def get_country_for_city(city: str) -> str:
    """
    Get country for a given city.
    
    Args:
        city: City name
        
    Returns:
        Country name or "Unknown" if not found
    """
    return CITY_TO_COUNTRY.get(city, "Unknown")


def get_flag_for_country(country: str) -> str:
    """
    Get flag emoji for a given country.
    
    Args:
        country: Country name
        
    Returns:
        Flag emoji or empty string if not found
    """
    return COUNTRY_FLAGS.get(country, "")


def get_country_with_flag(city: str) -> tuple:
    """
    Get country and flag for a given city.
    
    Args:
        city: City name
        
    Returns:
        Tuple of (country, flag_emoji)
    """
    country = get_country_for_city(city)
    flag = get_flag_for_country(country)
    return country, flag

