# Artifacts

An artifacts is a file or a set of files that are created during the training of a machine learning model. These artifacts can include the model itself, the training data, the validation data, the test data, the model's configuration, and the model's performance metrics.

## Dataset

### Cleanup Structure

| id | url | type | isMalicious |
| --- | --- | --- | --- |
| 1 | mp3raid.com/music/krizz_kaliko.html | benign | false |
| 2 | http://www.szabadmunkaero.hu/cimoldal.html?start=12  | defracment | true |
| 3 | br-icloud.com.br | phishing | true |

### Processed Structure 

id: id of url
url: url of website
type: type of url
use_of_ip: use of ip in url
abnormal_url: abnormal url
google_index: google search result
c_(content): count content in url
url_length: url length
hostname_length: hostname length
fd_length: fd length
tld_length: tld length
type_code: label encoding -> [benign: 0, defacement: 1, phishing: 2, ...]

| id | url | type | use_of_ip | abnormal_url | google_index | c_. | c_www | c_@ | c_dir | c_embed_domain | sus_url | short_url | c_https | c_http | c_% | c_? | c_ | c_ | url_length | hostname_length | fd_length | tld_length | c_digits | c_letters | type_code |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | br-icloud.com.br | phishing | 0 | 0 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 16 | 0 | 0 | -1 | 0 | 13 | 3 |
| 1 | mp3raid.com/music/krizz_kaliko.html | benign | 0 | 0 | 1 | 2 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 35 | 0 | 5 | -1 | 1 | 29 | 0 |
| 2 | http://www.lebensmittel-ueberwachung.de/index.php/aktuelles.1 | defacement | 0 | 1 | 1 | 4 | 1 | 0 | 2 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 61 | 32 | 9 | 2 | 1 | 50 | 1 |