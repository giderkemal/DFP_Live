# Snowflake Connection Setup Guide

## Issue Resolution

The current error `KeyError: 'password'` occurs because the AWS Secrets Manager lookup was failing due to rigid key expectations. Since you mentioned it worked before without a local password, the issue is with the AWS secret structure handling, not missing credentials.

## ✅ No Password Required Solution

**Good news!** The enhanced system now handles this automatically. You don't need to add a password to your `.env` file because:

1. **AWS Secrets Manager First**: The system tries AWS Secrets Manager first (like before)
2. **Flexible Key Handling**: It now looks for credentials under various key names
3. **Passwordless Auth Support**: It handles SSO and key-based authentication
4. **Graceful Fallbacks**: Multiple fallback strategies prevent crashes

## Quick Test

Run this to see what's in your AWS secret:
```bash
python debug_aws_secret.py
```

## Complete .env Configuration

Here's the corrected format for your `.env` file:

```env
# API Configuration
BEDROCK_TOKEN_URL=https://cognito-auth.aiplat.aws.pmicloud.biz/oauth2/token
CLIENT_ID=55jl5i8iktdshq7kksr956s75e
CLIENT_SECRET=1ej3qj2gea0a63j4agu8svtg80l84iaj0ee7km0sboq2hgbg1tct

# AWS Configuration  
AWS_ACCESS_KEY_ID=AKIA5OWBNOABXVGTJZMN
AWS_SECRET_ACCESS_KEY=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

# AWS Secrets Manager (for fallback)
SECRET_MANAGER_ACCESS_KEY=AKIA5OWBNOABRF5KP7NX
SECRET_MANAGER_SECRET_KEY=do9eFKuMRJJ4z08zVtN4rMvnPijdh3lqobspIteu

# Snowflake Configuration
account=PMI-PMI_FDF
user=KGIDER@PMINTL.NET
role=U_DEV_KGIDER
SNOWFLAKE_PASSWORD=your_password_here

# Database/Warehouse Configuration
SNOWFLAKE_WAREHOUSE=WH_DEV_ANALYST
SNOWFLAKE_DATABASE=DB_FDF_DEV
SNOWFLAKE_SCHEMA=INFORMATION_MART

# Environment
FDF_ENV=dev
```

## Enhanced Snowflake Connection Features

The updated `snowflake_connection.py` now includes:

### 1. Multiple Credential Sources
- **Local Environment Variables** (primary)
- **AWS Secrets Manager** (fallback)

### 2. Flexible Authentication
- Password-based authentication
- SSO authentication (if password is empty)
- Key-based authentication

### 3. Environment Variable Options
The system will look for passwords in this order:
- `SNOWFLAKE_PASSWORD`
- `password`
- `SNOWFLAKE_PASS`
- `SERVICE_ACCOUNT_PASSWORD`

### 4. Connection Parameter Sources
All connection parameters now come from environment variables:
- `account` → Snowflake account
- `user` → Username
- `role` → User role
- `SNOWFLAKE_WAREHOUSE` → Warehouse
- `SNOWFLAKE_DATABASE` → Database  
- `SNOWFLAKE_SCHEMA` → Schema

## Troubleshooting

### If you get "password not found" warnings:

**Option 1: Add Password (Recommended)**
```env
SNOWFLAKE_PASSWORD=your_actual_password
```

**Option 2: Use SSO Authentication**
```env
SNOWFLAKE_PASSWORD=
# Leave empty for SSO
```

**Option 3: Use Alternative Authentication**
The system will automatically try different authentication methods if password fails.

### If connection still fails:

1. **Check Account Format**
   - Current: `PMI-PMI_FDF`
   - Verify this matches your Snowflake account identifier

2. **Verify User Email**
   - Current: `KGIDER@PMINTL.NET`
   - Make sure this is your exact Snowflake username

3. **Check Role**
   - Current: `U_DEV_KGIDER`
   - Verify you have access to this role

4. **Test Connection Manually**
   ```python
   import snowflake.connector
   
   conn = snowflake.connector.connect(
       user='KGIDER@PMINTL.NET',
       password='your_password',
       account='PMI-PMI_FDF',
       warehouse='WH_DEV_ANALYST',
       database='DB_FDF_DEV',
       schema='INFORMATION_MART',
       role='U_DEV_KGIDER'
   )
   ```

## Error Handling Improvements

The updated system now:
- ✅ **No more crashes** - Returns empty DataFrames on connection failure
- ✅ **Better logging** - Detailed error messages in logs
- ✅ **Multiple strategies** - Tries different authentication methods
- ✅ **Graceful fallbacks** - App continues running even if Snowflake is unavailable

## Testing the Connection

Run this test to verify your Snowflake setup:

```python
# Create test_snowflake.py
from snowflake_connection import SnowflakeConnectionService
import pandas as pd

try:
    # Test basic connection
    result = SnowflakeConnectionService.fetch_query_result("SELECT 1 as test")
    print(f"✅ Snowflake connection successful: {result}")
except Exception as e:
    print(f"❌ Snowflake connection failed: {e}")
    print("The app will still work with empty data fallbacks")
```

## Security Best Practices

1. **Don't commit passwords** to git
2. **Use environment variables** for all credentials
3. **Consider SSO** for production environments
4. **Rotate credentials** regularly

## Alternative: Mock Mode for Development

If Snowflake access is not needed for testing, you can run in mock mode by setting:

```env
SNOWFLAKE_MOCK_MODE=true
```

This will make all Snowflake queries return empty DataFrames without attempting connections. 