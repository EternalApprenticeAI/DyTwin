SocialTwin Dataset
==================
Version : 1.0
Users   : 256
Rows    : 520,807 posts

Privacy Protection
------------------
- User identities anonymized via MD5 (filename = anonymous user ID)
- @mentions removed from all text fields (rule-based regex)
- URLs removed from all text fields
- Personal profile information (gender, region, bio, etc.) excluded

Columns
-------
原创/转发   : Post type (原创=original, 转发=retweet)
日期        : Post datetime
原微博内容  : Content of the original post being retweeted (cleaned)
全文内容    : User's retweet text (cleaned)
转发数      : Number of retweets this post received
评论数      : Number of comments
点赞数      : Number of likes
话题        : Hashtag topics (#tag#)
图文识别    : OCR content from post images (cleaned)
涉及行业    : Auto-labeled industry category
微博情绪    : Auto-labeled emotion category

Cleaning Stats
--------------
@mentions removed : 1,067,634
URLs removed      : 290,579

License
-------
For academic research use only. Do not attempt to re-identify users.

