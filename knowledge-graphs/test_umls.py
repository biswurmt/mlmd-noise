import requests

def test_umls_api_key(api_key):
    """
    Tests a UMLS API key by performing a basic search query.
    """
    print("Testing UMLS API key...")
    
    url = "https://uts-ws.nlm.nih.gov/rest/search/current"
    params = {
        "string": "Asthma",
        "apiKey": api_key
    }
    
    try:
        response = requests.get(url, params=params)
        
        # Check if the status code is 200 (OK)
        if response.status_code == 200:
            print("\n✅ Success! Your API key is valid and working.")
            
            # Extract and show a tiny bit of data to prove it worked
            results = response.json().get("result", {}).get("results", [])
            if results:
                print(f"Sample result found: {results[0]['name']} (UI: {results[0]['ui']})")
        
        elif response.status_code == 401:
            print("\n❌ Error 401: Unauthorized. Your API key is invalid or hasn't activated yet.")
            print(f"Response details: {response.text}")
            
        else:
            print(f"\n⚠️ Unexpected Status Code: {response.status_code}")
            print(f"Response details: {response.text}")
            
    except Exception as e:
        print(f"\n❌ A network or connection error occurred: {e}")

# --- RUN THE TEST ---
if __name__ == "__main__":
    # Replace the string below with your newly generated API key
    MY_NEW_API_KEY = "abfcd200-8513-4e3c-a70b-0359f7ea506d"
    
    test_umls_api_key(MY_NEW_API_KEY)