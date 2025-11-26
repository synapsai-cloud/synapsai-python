def build_url(base: str, endpoint: str) -> str:
    """
    Build a URL from a base URL and an endpoint.
    
    Args:
        base: Base URL.
        endpoint: Endpoint.
    
    Returns:
        URL.
    """
    base = base.rstrip("/")
    endpoint = endpoint.lstrip("/")
    return f"{base}/{endpoint}"