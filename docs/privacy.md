---
title: CORTEX privacy policy
---

# CORTEX privacy policy

*Last updated: 19 November 2026*

CORTEX is a personal tool used by one person, its developer, for their own job search. It is not offered to
the public, and no one else's Google account is connected to it.

## Gmail access

CORTEX asks for **read-only** access to Gmail (`gmail.readonly`). It can't send, delete, label or change any
email. It uses this access only to find replies to job applications (for example rejections or interview
invitations) in recent mail:

- Only recent messages are read. Promotions, social and forum mail is skipped.
- Messages are matched with fixed local rules. **Email content is never sent to any AI model or any other
  service.**
- **No email bodies and no sender addresses are stored.** CORTEX keeps only a message id, the matching result
  and a shortened subject line for the owner's own review.
- The access token is stored as a secret: in a private file on the owner's computer and in the encrypted secret
  store of the automation that runs CORTEX. It is never logged.

## Other data

CORTEX stores job postings, application statuses and notes in a private, password-protected database. This
data is not sold, shared or used for advertising.

## Google API user data

CORTEX's use of information received from Google APIs follows the
[Google API Services User Data Policy](https://developers.google.com/terms/api-services-user-data-policy),
including the Limited Use requirements.

## Removing access

Access can be revoked at any time at
[myaccount.google.com/permissions](https://myaccount.google.com/permissions).

## Contact

Questions: open an issue at [github.com/surya-teja0518/cortex](https://github.com/surya-teja0518/cortex/issues).
