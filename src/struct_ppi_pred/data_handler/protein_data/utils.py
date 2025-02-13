import requests
import json

UNIPROT_API_BASE_URL = "https://rest.uniprot.org/uniprotkb/"

def fetch_protein_data_uniprot(uniprot_id):
    """
    Fetches protein data from UniProt API for a given UniProt ID.

    Args:
        uniprot_id (str): UniProt Accession ID of the protein.
        fields (str, optional): Comma-separated list of fields to retrieve (e.g., "features,sequence,protein_names").
                                If None, fetches the full entry.

    Returns:
        dict or None: A dictionary containing protein data if successful,
                     None if there's an error.
    """
    url = f"{UNIPROT_API_BASE_URL}{uniprot_id}.json"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error for UniProt ID: {uniprot_id} - {http_err}")
        return None
    except requests.exceptions.RequestException as req_err:
        print(f"Request error for UniProt ID: {uniprot_id} - {req_err}")
        return None
    except json.JSONDecodeError as json_err:
        print(f"JSON decoding error for UniProt ID: {uniprot_id} - {json_err}")
        print(f"Response content: {response.text}")
        return None

def fetch_protein_strcture_af(uniprot_accession):
    """
    Makes a GET request to the AlphaFold API to retrieve structural predictions for a given UniProt Accession ID.

    Args:
        uniprot_accession (str): UniProt Accession ID of the protein to retrieve predictions for.

    Returns:
        dict or None: The JSON response from the API if successful, None if an error occurs.
    """
    api_endpoint = "https://alphafold.ebi.ac.uk/api/prediction/"
    url = f"{api_endpoint}{uniprot_accession}"  # Construct the URL for API

    try:
        # Use a timeout to handle potential connection issues
        response = requests.get(url, timeout=10)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            # Raise an exception for better error handling
            response.raise_for_status()
    except requests.exceptions.RequestException as e:
        pass