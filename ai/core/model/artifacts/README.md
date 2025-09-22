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

- id: id of url
- url: url of website
- type: type of url
- use_of_ip: use of ip in url
- abnormal_url: abnormal url
- google_index: google search result
- c_(content): count content in url
- url_length: url length
- hostname_length: hostname length
- fd_length: fd length
- tld_length: tld length
- type_code: label encoding -> [benign: 0, defacement: 1, phishing: 2, ...]

| ID | URL                                                                                                                                | Type       | IP Used | Abnormal | Google Index | `c_.` | `c_www` | `c_@` | `c_dir` | `c_embed_domain` | Suspicious | Short URL | HTTPS | HTTP | `%` | `?` | Extra c | URL Len | Host Len | FD Len | TLD Len | Digits | Letters | Type Code |
| -- | ---------------------------------------------------------------------------------------------------------------------------------- | ---------- | ------- | -------- | ------------ | ----- | ------- | ----- | ------- | ---------------- | ---------- | --------- | ----- | ---- | --- | --- | ------- | ------- | -------- | ------ | ------- | ------ | ------- | --------- |
| 0  | **br-icloud.com.br**                                                                                                               | Phishing   | 0       | 0        | 1            | 2     | 0       | 0     | 0       | 0                | 0          | 0         | 0     | 0    | 0   | 1   | 0       | 16      | 0        | 0      | -1      | 0      | 13      | 3         |
| 1  | **mp3raid.com/music/krizz\_kaliko.html**                                                                                           | Benign     | 0       | 0        | 1            | 2     | 0       | 0     | 2       | 0                | 0          | 0         | 0     | 0    | 0   | 0   | 0       | 35      | 0        | 5      | -1      | 1      | 29      | 0         |
| 2  | **[http://www.lebensmittel-ueberwachung.de/index.php/aktuelles.1](http://www.lebensmittel-ueberwachung.de/index.php/aktuelles.1)** | Defacement | 0       | 1        | 1            | 4     | 1       | 0     | 2       | 0                | 0          | 0         | 1     | 0    | 0   | 1   | 0       | 61      | 32       | 9      | 2       | 1      | 50      | 1         |

### Cross validation

- Cross valivation[https://medium.com/kbtg-life/what-is-cross-validation-cv-and-why-do-we-need-it-fb4bac340991]