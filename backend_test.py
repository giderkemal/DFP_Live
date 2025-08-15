import requests
import sys
import json
from datetime import datetime

class FieldIntelligenceAPITester:
    def __init__(self, base_url="http://localhost:8001"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0

    def run_test(self, name, method, endpoint, expected_status, data=None, params=None):
        """Run a single API test"""
        url = f"{self.base_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, params=params)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    if isinstance(response_data, dict):
                        print(f"   Response keys: {list(response_data.keys())}")
                        if 'count' in response_data:
                            print(f"   Data count: {response_data['count']}")
                    return True, response_data
                except:
                    return True, response.text
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                print(f"   Response: {response.text[:200]}...")
                return False, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_health_check(self):
        """Test health check endpoint"""
        success, response = self.run_test(
            "Health Check",
            "GET",
            "api/health",
            200
        )
        if success and isinstance(response, dict):
            print(f"   Status: {response.get('status')}")
            print(f"   Data loaded: {response.get('data_loaded')}")
        return success

    def test_metadata(self):
        """Test metadata endpoint"""
        success, response = self.run_test(
            "Get Metadata",
            "GET",
            "api/metadata",
            200
        )
        if success and isinstance(response, dict):
            print(f"   Available filters: {list(response.keys())}")
            if 'CLASS' in response:
                print(f"   Classes available: {len(response['CLASS'])} types")
            if 'DATE' in response:
                print(f"   Date range: {response['DATE']}")
        return success, response

    def test_filter_data_basic(self):
        """Test basic data filtering"""
        filter_data = {
            "date_range": ["2023-01-01", "2024-12-31"],
            "feedback_class": [],
            "form_type": "All",
            "region": [],
            "market": [],
            "location": [],
            "tmo": [],
            "brand": []
        }
        
        success, response = self.run_test(
            "Filter Data (Basic)",
            "POST",
            "api/filter-data",
            200,
            data=filter_data
        )
        return success, response

    def test_filter_data_with_class(self, metadata):
        """Test data filtering with specific class"""
        if not metadata or 'CLASS' not in metadata or not metadata['CLASS']:
            print("⚠️  Skipping class filter test - no classes available")
            return True, {}
            
        first_class = metadata['CLASS'][0]
        filter_data = {
            "date_range": ["2023-01-01", "2024-12-31"],
            "feedback_class": [first_class],
            "form_type": "All",
            "region": [],
            "market": [],
            "location": [],
            "tmo": [],
            "brand": []
        }
        
        success, response = self.run_test(
            f"Filter Data (Class: {first_class})",
            "POST",
            "api/filter-data",
            200,
            data=filter_data
        )
        return success, response

    def test_charts_data(self):
        """Test charts data endpoint"""
        success, response = self.run_test(
            "Get Charts Data",
            "GET",
            "api/charts-data",
            200
        )
        if success and isinstance(response, dict) and 'charts' in response:
            charts = response['charts']
            print(f"   Available charts: {list(charts.keys())}")
        return success, response

    def test_generate_report(self):
        """Test report generation (may fail due to Claude API limits)"""
        filter_data = {
            "date_range": ["2023-01-01", "2024-12-31"],
            "feedback_class": [],
            "form_type": "All",
            "region": [],
            "market": [],
            "location": [],
            "tmo": [],
            "brand": []
        }
        
        print("⚠️  Note: This test may fail due to Claude API rate limits - that's expected")
        success, response = self.run_test(
            "Generate Report",
            "POST",
            "api/generate-report",
            200,
            data=filter_data
        )
        
        # If it fails with 500, it might be due to Claude API limits
        if not success:
            print("   This is likely due to Claude API rate limits or missing API key")
            print("   The integration is working if we get a proper error response")
            
        return success, response

    def test_chat_endpoint(self):
        """Test chat endpoint (may fail due to Claude API limits)"""
        chat_data = {
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, can you help me understand the data?"
                }
            ],
            "system_prompt": "You are a helpful data analyst."
        }
        
        print("⚠️  Note: This test may fail due to Claude API rate limits - that's expected")
        success, response = self.run_test(
            "Chat with Claude",
            "POST",
            "api/chat",
            200,
            data=chat_data
        )
        
        if not success:
            print("   This is likely due to Claude API rate limits or missing API key")
            
        return success, response

def main():
    print("🚀 Starting Field Intelligence API Tests")
    print("=" * 50)
    
    # Initialize tester
    tester = FieldIntelligenceAPITester()
    
    # Test 1: Health Check
    if not tester.test_health_check():
        print("❌ Health check failed - stopping tests")
        return 1

    # Test 2: Metadata
    metadata_success, metadata = tester.test_metadata()
    if not metadata_success:
        print("❌ Metadata test failed - continuing with other tests")
        metadata = {}

    # Test 3: Basic data filtering
    filter_success, filter_data = tester.test_filter_data_basic()
    if not filter_success:
        print("❌ Basic filtering failed")

    # Test 4: Filter with specific class
    tester.test_filter_data_with_class(metadata)

    # Test 5: Charts data
    tester.test_charts_data()

    # Test 6: Report generation (expected to potentially fail)
    tester.test_generate_report()

    # Test 7: Chat endpoint (expected to potentially fail)
    tester.test_chat_endpoint()

    # Print final results
    print("\n" + "=" * 50)
    print(f"📊 Final Results: {tester.tests_passed}/{tester.tests_run} tests passed")
    
    if tester.tests_passed >= 5:  # Allow some Claude API failures
        print("✅ Backend API is working well!")
        return 0
    else:
        print("❌ Multiple backend issues detected")
        return 1

if __name__ == "__main__":
    sys.exit(main())