# Vouchers — 2026-05-27

Generated from `facts.voucher_set` / `facts.voucher_line`. Every figure traces to a page in a source PDF; the locator column names the page.

## Set totals

| Fund | Stated total | Parsed total | Lines | Held | Checks | Status | Register cross-check | Page |
|---|---:|---:|---:|---:|---:|---|---|---:|
| ACH | 9,394,987.52 | 9,394,987.52 | 2,189 | 0 | 407 | reconciled | yes — ties to the signed register | 54 |
| ASB | 181,039.41 | 181,039.41 | 316 | 0 | 100 | reconciled | yes — differs from the signed register | 8 |
| Capital | 1,047,509.82 | 1,047,509.82 | 39 | 0 | 27 | reconciled | yes — ties to the signed register | 1 |
| GF | 3,388,060.86 | 3,388,060.86 | 1,555 | 0 | 350 | reconciled | yes — differs from the signed register | 26 |
| Transportation | 173,922.13 | 173,922.13 | 1 | 0 | 1 | reconciled | **none recorded** | 1 |

**Total across sets that reconcile: 14,185,519.74**

Warrant periods: ACH 2026-04-09 → 2026-05-14; ASB 2026-04-09 → 2026-05-14; Capital 2026-04-09 → 2026-05-14; GF 2026-04-09 → 2026-05-14; Transportation 2026-03-13 → 2026-04-08.

P-card periods differ from warrant periods: ASB 2026-03-14 → 2026-04-24; Capital 2026-03-14 → 2026-04-24; GF 2026-03-14 → 2026-04-24.

Second control total (sum of deduplicated check amounts) differs from the invoice total on: ASB by 468.40. This is normal where a P-card statement total exceeds the transactions itemised, and where a credit rides on a sentinel check number.

## Checked against the board's signed register

| Fund | Basis | Register | Listing | Δ | Result | Warrant range |
|---|---|---:|---:|---:|---|---|
| ACH | ap_direct_deposit | 9,394,987.52 | 9,394,987.52 | 0.00 | MATCH | — |
| ASB | warrants_plus_pcard | 179,774.32 | 181,039.41 | 1,265.09 | REGISTER_MISMATCH | 418293-418299,418300-418313,418314-41832 |
| Capital | warrants_plus_pcard | 1,047,509.82 | 1,047,509.82 | 0.00 | MATCH | 208889-208891,208892-208895,208896-20890 |
| GF | warrants_plus_pcard | 3,383,444.97 | 3,388,060.86 | 4,615.89 | REGISTER_MISMATCH | 607932-607990,607991-608032,608033-60809 |

## Top 25 vendors this cycle

| Vendor | Invoice total | Of which new checks | Lines | Checks | From reconciled sets |
|---|---:|---:|---:|---:|---|
| GREEN RIVER COLLEGE | 2,988,834.00 | 2,988,834.00 | 1 | 1 | yes |
| Puget Sound Energy | 948,738.74 | 948,738.74 | 20 | 5 | yes |
| Sunburst Workforce Advisors LLC | 895,265.97 | 895,265.97 | 14 | 6 | yes |
| Pac West Mechanical LLC | 648,048.00 | 648,048.00 | 1 | 1 | yes |
| US Foods - Seattle | 449,211.70 | 449,211.70 | 24 | 6 | yes |
| Bank Of America | 428,682.04 | 428,682.04 | 1,222 | 15 | yes |
| Everdriven Technologies LLC | 379,989.98 | 379,989.98 | 18 | 5 | yes |
| AMAZON CAPITAL SERVICES | 347,949.01 | 347,949.01 | 1,194 | 6 | yes |
| Highline College | 341,053.45 | 341,053.45 | 2 | 2 | yes |
| First Student Inc | 313,731.10 | 313,731.10 | 21 | 5 | yes |
| GenCap Construction Corp | 310,898.08 | 310,898.08 | 1 | 1 | yes |
| Bellevue College | 258,963.34 | 258,963.34 | 1 | 1 | yes |
| KEA | 255,633.94 | 255,633.94 | 1 | 1 | yes |
| Comm in Schs of Kent | 236,185.60 | 236,185.60 | 6 | 3 | yes |
| City of Kent | 213,367.43 | 213,367.43 | 4 | 3 | yes |
| Elevation Healthcare LLC | 181,382.38 | 181,382.38 | 12 | 1 | yes |
| Schetky Northwest Sales Inc | 178,313.90 | 178,313.90 | 12 | 5 | yes |
| Brink Electric LLC | 175,578.00 | 175,578.00 | 2 | 1 | yes |
| CDW Government Inc | 150,808.05 | 150,808.05 | 3 | 1 | yes |
| Republic Svcs-#176/183 | 138,483.66 | 138,483.66 | 2 | 2 | yes |
| Eagle Asphalt Sealcoating Co LLC | 127,405.56 | 127,405.56 | 3 | 2 | yes |
| PetroCard Inc | 111,597.58 | 111,597.58 | 5 | 3 | yes |
| GERSH ACADEMY SEATTLE LLC | 106,300.64 | 106,300.64 | 4 | 1 | yes |
| WCP Sol Inc | 100,982.68 | 100,982.68 | 167 | 6 | yes |
| Renton SD | 100,451.16 | 100,451.16 | 8 | 2 | yes |

259 payee(s) totalling 809,142.75 are withheld from this export because they are not positively identifiable as organizations. They remain in `facts.voucher_line` and are queryable; they are not published because refunds and reimbursements to named individuals should not appear in a public briefing.

## Watch list, last 12 cycles

| Vendor | 04-23 | 05-28 | 06-25 | 07-23 | 08-13 | 09-24 | 10-22 | 12-10 | 01-14 | 02-11 | 03-25 | 05-27 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Sunburst Workforce Advisors | 610,158.65 | 584,420.74 | 392,944.59 | 1,249,322.98 | 278,180.58 | 26,191.50 | 35,486.75 | 1,012,574.33 | 398,849.69 | 970,495.51 | 722,422.23 | 895,265.97 |
| Elevation Healthcare | 24,854.25 | 53,335.50 | — | 37,816.13 | — | 45,525.32 | — | 38,181.33 | 48,085.50 | 197,885.38 | — | 181,382.38 |
| Blazerworks | 41,483.75 | 28,680.00 | 46,870.00 | 24,565.00 | 3,386.25 | 153,034.27 | — | 60,884.50 | 155,824.22 | 61,566.30 | 105,425.90 | 40,500.25 |
| CBPI | 7,335.70 | 11,095.40 | 9,467.10 | 11,555.60 | 6,931.80 | — | — | 11,782.85 | 20,094.40 | 6,948.75 | 12,303.75 | — |
| Positive Behavior Supports | — | 22,398.75 | 10,822.50 | 13,162.50 | 8,235.00 | — | — | — | — | — | — | — |
| Robert Half | — | — | — | — | 23,312.00 | 12,152.00 | 9,920.00 | 12,400.00 | 22,320.00 | 7,477.20 | 12,586.00 | 15,888.57 |
| Pacifica Law Group | 38,898.44 | 13,594.00 | 38,073.00 | — | 48,453.50 | 79,537.85 | 73,021.05 | 41,649.50 | 51,466.50 | 73,095.00 | 49,461.50 | 65,126.50 |
| Foster Garvey | 17,843.62 | 20,438.34 | — | 40,533.55 | 27,185.31 | 38,916.01 | 52,818.89 | 7,064.91 | 251.73 | — | — | — |
| KCABA | 135,993.75 | 136,143.75 | 114,712.50 | 27,375.00 | 187,931.25 | 25,126.35 | — | 166,031.25 | 255,168.75 | 63,262.50 | 144,656.25 | — |
| Gersh Academy | 74,290.72 | 88,128.90 | 83,354.72 | 102,706.77 | 78,866.78 | 213,993.10 | 53,327.50 | 91,236.50 | 185,748.46 | 71,464.16 | 90,329.66 | 106,300.64 |
| Renton SD | 54,749.41 | 68,297.09 | 60,498.63 | 71,110.05 | 67,312.81 | 59,547.51 | — | 92,569.99 | 169,155.57 | 71,640.05 | 92,105.18 | 100,451.16 |
| Tacoma SD | 6,046.44 | 14,285.58 | — | — | 18,810.05 | 291,290.14 | 4,900.11 | 5,939.92 | 149,140.20 | 9,421.58 | 9,878.56 | — |
| Yellow Wood Academy | — | 97,411.00 | — | 51,504.67 | 10,077.00 | 98,968.27 | — | 49,848.40 | 99,696.80 | 49,848.40 | 49,848.40 | 39,166.60 |
| Hazel Health | 25,000.00 | 25,000.00 | 25,000.00 | 25,000.00 | — | 79,166.67 | 25,000.00 | — | 75,000.00 | 25,000.00 | 25,000.00 | 25,000.00 |
| St Vincent de Paul | 28,966.67 | 28,966.67 | 28,966.67 | 28,966.67 | 28,966.67 | 28,966.67 | 28,966.67 | 28,966.67 | 57,933.34 | 28,966.67 | 28,966.67 | 28,966.67 |

Blank cells mean the vendor does not appear in that cycle's listings, not that it was paid nothing outside them.

---

Records retention: DAN GS2011-184 (Washington State Archives). Source files are named in `facts.voucher_set.locator_file_path`.
