# SocialTwin Data

Place the full SocialTwin user CSV files in this directory:

```text
data/SocialTwin/
```

Each user is stored as one anonymized CSV file, for example:

```text
data/SocialTwin/0eced5cf.csv
```

Expected columns include:

| Column | Description |
| --- | --- |
| `原创/转发` | Post type, original or repost |
| `日期` | Timestamp |
| `原微博内容` | Original post content for repost records |
| `全文内容` | User-authored content or repost text |
| `转发数` | Repost count |
| `评论数` | Comment count |
| `点赞数` | Like count |
| `话题` | Topic tags |
| `图文识别` | OCR or image-text recognition content |
| `涉及行业` | Industry/category label |
| `微博情绪` | Sentiment label |

The full dataset is not committed to GitHub because it is large. 