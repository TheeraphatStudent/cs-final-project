มี idea หนึ่ง เราจะ train model ประมาณนี้

https://medium.com/sfu-cspmp/detecting-malicious-urls-2412091872d6

แต่จะไม่เอาข้อมูล url มาเทรน แต่จะเป็น keywords พวก web พนัน

จะอารามประมาณนี้ https://github.com/ani10030/bad-words-detector/tree/master

แล้วทำ server เป็น web api ฝั่ง client อาจเขียนเป็น mobile/web app หรือ chrome extension ไปดึงข้อมูลจาก api ของ ai ที่เราเทรนมาเช็คว่าเป็นเว็บพนันไหม หากใช่ก็เตือนผู้ใช้

---

## Client

ดาวน์โหลด Chrome Extension เพื่อใช้งาน

## Server

ประมวลผล URL ที่ User ได้รับ

---

## Flowchart

[Flow ของระบบ](https://miro.com/welcomeonboard/YkJvUDk3ZGswRW03aVlDV3J6VUVEL3lPZWg3NUE2bllzNVhzelMwcDFRa3laZWI1V3NMTVhadjVYelNFTUppSjFPNWJXYVl2T3dDRTRMSFV1eFk5Y0R3OW1VWXNxaURVZDROUHk5S1FQQ2xWQnYvU0dCQlNSdjFaWVBhVzFlazNnbHpza3F6REdEcmNpNEFOMmJXWXBBPT0hdjE=?share_link_id=867774107315)