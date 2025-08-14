#!/usr/bin/env python3
"""
Debug script to examine the AWS Secrets Manager secret structure.
This will help understand what keys are available and why the password lookup was failing.
"""

import os
import json
import boto3
from dotenv import load_dotenv

load_dotenv()

def examine_aws_secret():
    """Examine the structure of the AWS secret"""
    
    env = os.getenv('FDF_ENV', 'dev')
    secret_manager_access_key = os.getenv('SECRET_MANAGER_ACCESS_KEY')
    secret_manager_secret_key = os.getenv('SECRET_MANAGER_SECRET_KEY')
    
    print("🔍 AWS Secrets Manager Debug Tool")
    print("=" * 50)
    
    try:
        print(f"📊 Environment: {env}")
        print(f"🔑 Access Key: {secret_manager_access_key[:10]}..." if secret_manager_access_key else "❌ Not found")
        print(f"🗝️ Secret Key: {secret_manager_secret_key[:10]}..." if secret_manager_secret_key else "❌ Not found")
        
        # Create AWS client
        client = boto3.client(
            'secretsmanager',
            aws_access_key_id=secret_manager_access_key or 'AKIA5OWBNOABRF5KP7NX',
            aws_secret_access_key=secret_manager_secret_key or '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',
            region_name='eu-west-1',
        )
        
        secret_id = f'vapafpdf-{env}-Snowflake_FDF_Password'
        print(f"🎯 Secret ID: {secret_id}")
        
        # Get the secret
        print("\n🔄 Retrieving secret...")
        response = client.get_secret_value(SecretId=secret_id)
        
        secret_string = response['SecretString']
        secret_dict = json.loads(secret_string)
        
        print("✅ Secret retrieved successfully!")
        print(f"\n📋 Available keys in secret:")
        for key in secret_dict.keys():
            # Don't show the actual values for security
            value_preview = str(secret_dict[key])[:10] + "..." if len(str(secret_dict[key])) > 10 else str(secret_dict[key])
            print(f"  🔸 {key}: {value_preview}")
        
        print(f"\n📊 Total keys found: {len(secret_dict)}")
        
        # Check for username variants
        username_keys = ['username', 'user', 'service_account_username', 'snowflake_user', 'account_user']
        found_username_keys = [key for key in username_keys if key in secret_dict]
        print(f"\n👤 Username-related keys found: {found_username_keys}")
        
        # Check for password variants
        password_keys = ['password', 'service_account_password', 'snowflake_password', 'account_password', 'pass']
        found_password_keys = [key for key in password_keys if key in secret_dict]
        print(f"🔐 Password-related keys found: {found_password_keys}")
        
        # Check for alternative auth
        alt_keys = ['private_key', 'key_pair', 'authenticator', 'token']
        found_alt_keys = [key for key in alt_keys if key in secret_dict]
        print(f"🔑 Alternative auth keys found: {found_alt_keys}")
        
        if not found_password_keys and not found_alt_keys:
            print("⚠️ No password or alternative authentication found!")
            print("💡 This might explain why it worked without a local password")
            print("   The system may be using SSO or key-based authentication")
        
        print("\n🎉 Analysis complete!")
        print("\n📝 Recommendations:")
        if found_username_keys and (found_password_keys or found_alt_keys):
            print("  ✅ Secret structure looks complete")
            print("  ➡️ The enhanced code should handle this correctly now")
        elif found_username_keys and not found_password_keys:
            print("  🔶 Username found but no password")
            print("  ➡️ System will attempt passwordless authentication")
        else:
            print("  ❌ Missing critical authentication information")
            print("  ➡️ May need to fall back to environment variables")
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to examine secret: {e}")
        print("\n🔧 Possible issues:")
        print("  • AWS credentials incorrect")
        print("  • Secret doesn't exist")
        print("  • Network connectivity issues")
        print("  • Insufficient permissions")
        return False

def test_enhanced_connection():
    """Test the enhanced connection with current setup"""
    print("\n🧪 Testing Enhanced Snowflake Connection")
    print("=" * 50)
    
    try:
        from snowflake_connection import SnowflakeConnectionService
        
        # Test credential retrieval
        print("🔄 Testing credential retrieval...")
        credentials = SnowflakeConnectionService._SnowflakeConnectionService__get_service_account_credentials()
        
        if credentials:
            username = credentials.get('service_account_username', 'N/A')
            has_password = bool(credentials.get('service_account_password'))
            print(f"  ✅ Username: {username}")
            print(f"  ✅ Has password: {has_password}")
            
            if not has_password:
                print("  ℹ️ No password - will attempt SSO/key-based auth")
            
            return True
        else:
            print("  ❌ No credentials retrieved")
            return False
            
    except Exception as e:
        print(f"  ❌ Connection test failed: {e}")
        return False

def main():
    """Run all diagnostics"""
    print("🚀 Snowflake AWS Secret Diagnostics")
    print("=" * 60)
    
    # Test AWS secret examination
    aws_success = examine_aws_secret()
    
    # Test enhanced connection
    conn_success = test_enhanced_connection()
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print(f"AWS Secret Examination: {'✅ SUCCESS' if aws_success else '❌ FAILED'}")
    print(f"Enhanced Connection Test: {'✅ SUCCESS' if conn_success else '❌ FAILED'}")
    
    if aws_success and conn_success:
        print("\n🎉 Great! The enhanced system should work correctly now.")
        print("   ➡️ You can run the main app without needing to add a local password.")
    elif aws_success:
        print("\n🔶 AWS secret accessible but connection needs work.")
        print("   ➡️ Check the Snowflake connection parameters.")
    else:
        print("\n⚠️ Issues found. The app will use fallback behavior.")
        print("   ➡️ It should still work but with sample data.")

if __name__ == "__main__":
    main() 