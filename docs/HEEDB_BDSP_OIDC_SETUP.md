# HEEDB external diagnostic validation — AWS/BDSP binding

This repository must not store long-lived AWS access keys for HEEDB.

Use GitHub OIDC to assume a read-only IAM role in the AWS account that BDSP has
authorized. BDSP exposes the credentialed ECG repository through an S3 Access
Point alias shown in the BDSP Cloud Credentials dashboard.

## Required GitHub repository variables

Create these **Actions variables** (not secrets):

- `BDSP_AWS_ROLE_ARN` — ARN of the read-only IAM role GitHub Actions may assume.
- `BDSP_S3_ACCESS_POINT_ALIAS` — exact credentialed-access S3 alias shown by BDSP.
- `BDSP_AWS_REGION` — normally `us-east-1`.

Do not commit AWS access keys, secret keys, session tokens, or BDSP credentials.

## GitHub OIDC provider

Provider URL:

`https://token.actions.githubusercontent.com`

Audience:

`sts.amazonaws.com`

## IAM trust policy

Replace `<AWS_ACCOUNT_ID>` and `<ROLE_NAME>` as appropriate. Restrict the
role to the MEDCALC repository's main branch.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::<AWS_ACCOUNT_ID>:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
          "token.actions.githubusercontent.com:sub": "repo:comincinieric56-maker/medcalc-clinico:ref:refs/heads/main"
        }
      }
    }
  ]
}
```

If the AWS account has opted into GitHub's newer customized OIDC subject format,
use the exact `sub` claim shown by AWS/GitHub rather than weakening the trust
policy.

## Read-only role permissions

The role needs only list/read operations. BDSP's Access Point policy remains the
external authorization boundary.

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": "*"
    }
  ]
}
```

When the exact Access Point ARN is available, this policy should be narrowed to
that Access Point and its object ARN rather than `*`.

## Safety sequence

1. Preflight only: assume the role, verify the AWS account identity, and list
   the top-level `ECG/` prefix.
2. Download only I0001/I0006 `metadata.csv` files.
3. Freeze the 10,000-patient label-blind cohort.
4. Run all MEDCALC predictions.
5. Only after prediction completion, open
   `12SL_diagnoses/diagnoses_acquisition.csv` and
   `diagnoses_dictionary.csv`.
6. Score physician-overread labels using the already-frozen mapping contract.
7. Persist only aggregate metrics. Never persist row-level gold/prediction joins
   in GitHub artifacts or the repository.
8. Mark HEEDB consumed immediately after first scoring.

## External action provenance

The workflow uses `aws-actions/configure-aws-credentials` pinned to immutable
commit `e1253824e5c10ff9df46874f81ed3ec929e19cfd`
(`v6.3.0`, MIT license).
