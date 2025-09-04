import joblib
import re
from urllib.parse import urlparse
from tld import get_tld
import os
import pandas as pd

# Load the trained model
# This makes the path relative to the script's location
model_path = os.path.join(os.path.dirname(__file__), "..", "model", "code", "exports", "models")
model = joblib.load(os.path.join(model_path, "best_svm_model.pkl"))
scaler = joblib.load(os.path.join(model_path, "svm_scaler.pkl"))

# --- Feature Extraction Functions ---

def having_ip_address(url: str) -> int:
    """Checks if the URL contains an IP address."""
    match = re.search(
        r'(([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])\.'
        r'([01]?\d\d?|2[0-4]\d|25[0-5])\/)|'  # IPv4
        r'((0x[0-9a-fA-F]{1,2})\.(0x[0-9a-fA-F]{1,2})\.(0x[0-9a-fA-F]{1,2})\.(0x[0-9a-fA-F]{1,2})\/)|'  # IPv4 in hexadecimal
        r'(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # IPv6
    return 1 if match else 0

def abnormal_url(url: str) -> int:
    """Checks if the hostname is present in the URL path, which can be a sign of obfuscation."""
    hostname = urlparse(url).hostname
    if hostname:
        match = re.search(hostname, url)
        return 1 if match else 0
    return 0

def count_dot(url: str) -> int:
    """Counts the number of dots in the URL."""
    return url.count('.')

def count_www(url: str) -> int:
    """Counts the occurrences of 'www' in the URL."""
    return url.count('www')

def count_at(url: str) -> int:
    """Counts the number of '@' symbols in the URL."""
    return url.count('@')

def no_of_dir(url: str) -> int:
    """Counts the number of directories in the URL path."""
    urldir = urlparse(url).path
    return urldir.count('/')

def no_of_embed(url: str) -> int:
    """Counts the number of embedded domains in the URL path."""
    urldir = urlparse(url).path
    return urldir.count('//')

def shortening_service(url: str) -> int:
    """Checks if the URL uses a known shortening service."""
    match = re.search(r'bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      r'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|zpr\.io|'
                      r'alturl\.com|url4\.eu|'
                      r'bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      r'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|'
                      r'buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|'
                      r'prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|'
                      r'qr\.net|1url\.com|tweez\.me|v\.gd|tr\.im|link\.zip\.net',
                      url)
    return 1 if match else 0

def count_percent(url: str) -> int:
    """Counts the number of '%' symbols."""
    return url.count('%')

def count_question(url: str) -> int:
    """Counts the number of '?' symbols."""
    return url.count('?')

def count_hyphen(url: str) -> int:
    """Counts the number of '-' symbols."""
    return url.count('-')

def count_equal(url: str) -> int:
    """Counts the number of '=' symbols."""
    return url.count('=')

def url_length(url: str) -> int:
    """Returns the length of the URL."""
    return len(url)

def count_https(url: str) -> int:
    """Counts the occurrences of 'https' in the URL."""
    return url.count('https')

def count_http(url: str) -> int:
    """Counts the occurrences of 'http' in the URL."""
    return url.count('http')

def hostname_length(url: str) -> int:
    """Returns the length of the hostname."""
    return len(urlparse(url).netloc)

def suspicious_words(url: str) -> int:
    """Checks for suspicious words in the URL."""
    match = re.search(r'PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr',
                      url)
    return 1 if match else 0

def fd_length(url: str) -> int:
    """Returns the length of the first directory in the path."""
    urlpath = urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except IndexError:
        return 0

def tld_length(url: str) -> int:
    """Calculates the length of the top-level domain."""
    try:
        tld_obj = get_tld(url, as_object=True, fail_silently=True)
        if tld_obj:
            # The notebook calculates the length of the TLD (e.g., 'com')
            return len(tld_obj.tld)
        return -1
    except Exception:
        return -1

def digit_count(url: str) -> int:
    """Counts the number of digits in the URL."""
    return sum(c.isdigit() for c in url)

def letter_count(url: str) -> int:
    """Counts the number of letters in the URL."""
    return sum(c.isalpha() for c in url)


def predict_url(url: str, is_test_benign: bool = False) -> str:
    """
    Predicts if a URL is malicious or not by extracting 21 features.
    """
    # Feature order must match the order used during training
    features = [
        having_ip_address(url),      # use_of_ip
        abnormal_url(url),           # abnormal_url
        count_dot(url),              # count_.
        count_www(url),              # count_www
        count_at(url),               # count_@
        no_of_dir(url),              # count_dir
        no_of_embed(url),            # count_embed_domain
        shortening_service(url),     # short_url
        count_percent(url),          # count%
        count_question(url),         # count?
        count_hyphen(url),           # count-
        count_equal(url),            # count=
        url_length(url),             # url_length
        count_https(url),            # count_https
        count_http(url),             # count_http
        hostname_length(url),        # hostname_length
        suspicious_words(url),       # sus_url
        fd_length(url),              # fd_length
        tld_length(url),             # tld_length
        digit_count(url),            # count_digits
        letter_count(url)            # count_letters
    ]
    
    # Create a DataFrame with the correct feature names
    feature_names = ['use_of_ip','abnormal_url', 'count_.', 'count_www', 'count_@',
                     'count_dir', 'count_embed_domain', 'short_url', 'count%', 'count?',
                     'count-', 'count=', 'url_length', 'count_https', 'count_http',
                     'hostname_length', 'sus_url', 'fd_length', 'tld_length', 'count_digits',
                     'count_letters']
    
    df = pd.DataFrame([features], columns=feature_names)

    if is_test_benign:
        df['abnormal_url'] = 0

    # Scale the features using the DataFrame
    scaled_features = scaler.transform(df)

    # Define the label mapping
    label_mapping = {
        0: "Benign",
        1: "Defacement",
        2: "Malware",
        3: "Phishing"
    }

    print(f"\n--- Debugging Features for URL: {url} ---")
    print(df.to_string(header=True, index=False))

    # Make the prediction and return the corresponding label
    prediction_code = model.predict(scaled_features)[0]
    return label_mapping.get(prediction_code, "Unknown")


if __name__ == '__main__':
    # Example Usage:
    test_url_benign = "http://www.google.com"
    result_benign = predict_url(test_url_benign, is_test_benign=True)
    print(f"The URL '{test_url_benign}' is predicted to be: {result_benign}")

    test_url_malicious = "http://133.133.133.133/bad-site"
    result_malicious = predict_url(test_url_malicious)
    print(f"The URL '{test_url_malicious}' is predicted to be: {result_malicious}")
