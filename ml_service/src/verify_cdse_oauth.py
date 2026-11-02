"""
GeoGuard CDSE OAuth2 Authentication Verification Script.

Tests:
1. Loading credentials securely from ml_service/.env
2. Requesting OAuth2 token via client_credentials flow from CDSE Keycloak
3. Verifying access token generation without logging secrets or tokens
4. Testing an authenticated request against a real Sentinel-2 L2A product:
   - Querying catalogue for active Sentinel-2 L2A product metadata
   - Testing authenticated access against Sentinel Hub Processing API
"""

import os
import sys
import requests

current_dir = os.path.dirname(os.path.abspath(__file__))
ml_service_dir = os.path.dirname(current_dir)
env_path = os.path.join(ml_service_dir, ".env")

if not os.path.exists(env_path):
    print(f"Error: {env_path} does not exist.")
    sys.exit(1)

# Parse .env
env = {}
with open(env_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip("'\"")

client_id = env.get("CDSE_CLIENT_ID")
client_secret = env.get("CDSE_CLIENT_SECRET")

print("==================================================================")
print("1. Credential Ingestion Verification")
print("==================================================================")
print(f"Loaded credentials from: {env_path}")
print(f"Client ID configured: {bool(client_id)} (Character count: {len(client_id) if client_id else 0})")
print(f"Client Secret configured: {bool(client_secret)} (Character count: {len(client_secret) if client_secret else 0})")

token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
payload = {
    "grant_type": "client_credentials",
    "client_id": client_id,
    "client_secret": client_secret,
}

print("\n==================================================================")
print("2. OAuth2 Token Request (client_credentials)")
print("==================================================================")
print(f"Target URL: {token_url}")
try:
    resp = requests.post(token_url, data=payload, timeout=25)
    print(f"HTTP Status: {resp.status_code}")
except Exception as e:
    print(f"Connection error to token service: {e}")
    sys.exit(1)

if resp.status_code != 200:
    print("OAuth2 Authentication Status: FAILED")
    print(f"Response: {resp.text[:300]}")
    sys.exit(1)

data = resp.json()
token = data.get("access_token")
expires_in = data.get("expires_in")
token_type = data.get("token_type")

print("OAuth2 Authentication Status: SUCCESS")
print(f"Token Type: {token_type}")
print(f"Token Lifetime: {expires_in} seconds ({expires_in // 60} minutes)")
print("Token Scope:", data.get("scope"))
print("Access Token: Successfully acquired (Value suppressed for security)")

print("\n==================================================================")
print("3. Authenticated Access to Real Sentinel-2 L2A Product")
print("==================================================================")

headers = {"Authorization": f"Bearer {token}"}

# Test A: Query Real Sentinel-2 L2A Product Metadata via CDSE OData
cat_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?$filter=startswith(Name,'S2B_MSIL2A_')&$top=1"
print(f"A. Querying OData Catalogue with Bearer Token...")
r_cat = requests.get(cat_url, headers=headers, timeout=25)
print(f"   Catalogue Query HTTP Status: {r_cat.status_code}")

if r_cat.status_code == 200:
    val = r_cat.json().get("value", [])
    if val:
        prod = val[0]
        prod_id = prod.get("Id")
        prod_name = prod.get("Name")
        content_len = prod.get("ContentLength", 0)
        origin_date = prod.get("OriginDate")
        print(f"   Discovered Product Name: {prod_name}")
        print(f"   Product ID: {prod_id}")
        print(f"   Product Archive Size: {content_len / (1024*1024):.2f} MB")
        print(f"   Origin Date: {origin_date}")

# Test B: Test Sentinel Hub Processing API on Copernicus Data Space
print(f"\nB. Testing Sentinel Hub Processing API with Bearer Token...")
sh_url = "https://sh.dataspace.copernicus.eu/api/v1/process"

# Minimal processing body for Sentinel-2 L2A over sample test bounds
evalscript = """
//VERSION=3
function setup() {
  return {
    input: ["B02", "B03", "B04", "B08"],
    output: { bands: 4, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(sample) {
  return [sample.B02, sample.B03, sample.B04, sample.B08];
}
"""
sh_body = {
    "input": {
        "bounds": {
            "bbox": [76.95, 12.00, 77.05, 12.10],
            "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
        },
        "data": [{"type": "sentinel-2-l2a", "dataFilter": {"maxCloudCoverage": 30.0}}],
    },
    "output": {
        "width": 128,
        "height": 128,
        "responses": [{"format": {"type": "image/tiff"}}],
    },
    "evalscript": evalscript,
}

sh_headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}

try:
    r_sh = requests.post(sh_url, headers=sh_headers, json=sh_body, timeout=30)
    print(f"   Sentinel Hub Process API HTTP Status: {r_sh.status_code}")
    if r_sh.status_code == 200:
        print(f"   Sentinel Hub Real Data Retrieval: SUCCESS!")
        print(f"   Retrieved TIFF raster: {len(r_sh.content)} bytes of real 4-band Sentinel-2 imagery!")
        import io, tifffile
        with io.BytesIO(r_sh.content) as buf:
            raster_arr = tifffile.imread(buf)
        print(f"   Raster Shape: {raster_arr.shape} (Height, Width, Channels)")
        print(f"   Data Type: {raster_arr.dtype}")
        print(f"   Channel 0 (B02) range: [{raster_arr[:,:,0].min():.4f}, {raster_arr[:,:,0].max():.4f}]")
        print(f"   Channel 1 (B03) range: [{raster_arr[:,:,1].min():.4f}, {raster_arr[:,:,1].max():.4f}]")
        print(f"   Channel 2 (B04) range: [{raster_arr[:,:,2].min():.4f}, {raster_arr[:,:,2].max():.4f}]")
        print(f"   Channel 3 (B08) range: [{raster_arr[:,:,3].min():.4f}, {raster_arr[:,:,3].max():.4f}]")
    else:
        print(f"   Sentinel Hub Process API Response: HTTP {r_sh.status_code} - {r_sh.text[:300]}")
except Exception as e:
    print(f"   Exception contacting Sentinel Hub: {e}")
