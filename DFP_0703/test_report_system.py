#!/usr/bin/env python3
"""
Test script for the improved report template system.
Run this to verify the enhancements work correctly on a new deployment.
"""

import pandas as pd
import sys
import traceback

def test_xml_processing():
    """Test the XMLDataProcessor functionality"""
    print("🧪 Testing XML Processing...")
    
    try:
        from models import XMLDataProcessor
        
        # Create test data with various edge cases
        test_df = pd.DataFrame({
            'field_intelligence_translated': [
                'Sample feedback with special chars: <>&"\'',
                'Very long text that should be truncated if needed ' * 100,
                None,  # Test None handling
                ''     # Test empty string
            ],
            'submission_datetime': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
            'class': ['Issue A', 'Issue B', None, 'Issue D'],
            'location_name': ['Location 1', None, 'Location 3', ''],
            'form_type': ['CONSUMER_FEEDBACK', 'BRAND_SOURCING', 'CROSS_CATEGORY', 'TOBACCO_CATEGORY']
        })
        
        # Test XML generation
        xml_result = XMLDataProcessor.create_xml_infos(test_df)
        
        if xml_result and len(xml_result) > 50:
            print(f"  ✅ XML generated successfully: {len(xml_result)} characters")
            
            # Verify XML is well-formed
            import xml.etree.ElementTree as ET
            try:
                ET.fromstring(xml_result)
                print("  ✅ XML is well-formed")
            except ET.ParseError as e:
                print(f"  ❌ XML parsing failed: {e}")
                return False
                
        else:
            print(f"  ❌ XML generation failed or too short: {len(xml_result) if xml_result else 0}")
            return False
            
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        xml_empty = XMLDataProcessor.create_xml_infos(empty_df)
        if xml_empty:
            print("  ✅ Empty DataFrame handled gracefully")
        else:
            print("  ❌ Empty DataFrame not handled properly")
            return False
            
        return True
        
    except Exception as e:
        print(f"  ❌ XML processing test failed: {e}")
        traceback.print_exc()
        return False

def test_template_manager():
    """Test the ReportTemplateManager functionality"""
    print("\n🧪 Testing Template Manager...")
    
    try:
        from models import ReportTemplateManager
        
        # Create test data
        test_df = pd.DataFrame({
            'field_intelligence_translated': ['Sample feedback 1', 'Sample feedback 2'],
            'submission_datetime': ['2024-01-01', '2024-01-02'],
            'class': ['Issue A', 'Issue B'],
            'location_name': ['Location 1', 'Location 2'],
            'form_type': ['CONSUMER_FEEDBACK', 'BRAND_SOURCING'],
            'product_category_name': ['Product A', 'Product B'],
            'brand_name_from': ['Brand X', 'Brand Y'],
            'brand_name_to': ['Brand Z', 'Brand W'],
            'pmi_product_name': ['PMI Product 1', 'PMI Product 2'],
            'tmo_name': ['TMO 1', 'TMO 2']
        })
        
        # Test generic template
        prompt_generic, success_generic = ReportTemplateManager.create_structured_prompt(
            test_df, template_name="generic"
        )
        
        if success_generic and prompt_generic and len(prompt_generic) > 500:
            print(f"  ✅ Generic template: Success, {len(prompt_generic)} characters")
        else:
            print(f"  ❌ Generic template failed: Success={success_generic}, Length={len(prompt_generic) if prompt_generic else 0}")
            return False
            
        # Test comprehensive template
        prompt_comp, success_comp = ReportTemplateManager.create_structured_prompt(
            test_df, template_name="comprehensive"
        )
        
        if success_comp and prompt_comp and len(prompt_comp) > 500:
            print(f"  ✅ Comprehensive template: Success, {len(prompt_comp)} characters")
        else:
            print(f"  ❌ Comprehensive template failed: Success={success_comp}, Length={len(prompt_comp) if prompt_comp else 0}")
            return False
            
        # Test with extra parameters
        prompt_extra, success_extra = ReportTemplateManager.create_structured_prompt(
            test_df, 
            template_name="generic",
            extra_demand="Focus on seasonal trends",
            extra_instructions="Include detailed metrics"
        )
        
        if success_extra and "seasonal trends" in prompt_extra and "detailed metrics" in prompt_extra:
            print("  ✅ Extra parameters handled correctly")
        else:
            print("  ❌ Extra parameters not handled properly")
            return False
            
        return True
        
    except Exception as e:
        print(f"  ❌ Template manager test failed: {e}")
        traceback.print_exc()
        return False

def test_report_extractor():
    """Test the ReportExtractor functionality"""
    print("\n🧪 Testing Report Extractor...")
    
    try:
        from models import ReportExtractor
        
        # Test case 1: Well-formed XML
        test_xml_good = '''
        <response>
            <report>
            This is a test report with multiple lines.
            
            It contains operational challenges and insights.
            Some specific examples with [Row_ID:123] and [Row_ID:456].
            </report>
            <examplesID>123, 456, 789</examplesID>
        </response>
        '''
        
        report, report_success = ReportExtractor.extract_report_from_output(test_xml_good)
        examples, examples_success = ReportExtractor.extract_examples_id_from_output(test_xml_good)
        
        if report_success and "operational challenges" in report:
            print("  ✅ Well-formed XML extraction successful")
        else:
            print(f"  ❌ Well-formed XML extraction failed: Success={report_success}")
            return False
            
        if examples_success and len(examples) >= 3:
            print(f"  ✅ Example IDs extracted: {examples}")
        else:
            print(f"  ❌ Example IDs extraction failed: Success={examples_success}, Count={len(examples) if examples else 0}")
            return False
            
        # Test case 2: Malformed XML (missing closing tags)
        test_xml_bad = '''
        <response>
            <report>
            This is a malformed report without proper closing
            <examplesID>123, 456
        '''
        
        report_bad, success_bad = ReportExtractor.extract_report_from_output(test_xml_bad)
        
        if report_bad and len(report_bad) > 10:  # Should still extract something
            print("  ✅ Malformed XML handled gracefully")
        else:
            print(f"  ❌ Malformed XML not handled properly: {report_bad}")
            return False
            
        # Test case 3: No XML structure at all
        test_no_xml = '''
        This is just plain text with no XML structure.
        Problem 1: Inventory issues at location A
        Problem 2: Pricing concerns for Brand X
        [Row_ID:100] and [Row_ID:200] were mentioned.
        '''
        
        report_plain, success_plain = ReportExtractor.extract_report_from_output(test_no_xml)
        examples_plain, ex_success_plain = ReportExtractor.extract_examples_id_from_output(test_no_xml)
        
        if report_plain and len(report_plain) > 10:
            print("  ✅ Plain text fallback working")
        else:
            print(f"  ❌ Plain text fallback failed: {report_plain}")
            return False
            
        if examples_plain and len(examples_plain) >= 2:
            print(f"  ✅ Pattern-based ID extraction: {examples_plain}")
        else:
            print(f"  ❌ Pattern-based ID extraction failed: {examples_plain}")
            return False
            
        return True
        
    except Exception as e:
        print(f"  ❌ Report extractor test failed: {e}")
        traceback.print_exc()
        return False

def test_backward_compatibility():
    """Test that the old interface still works"""
    print("\n🧪 Testing Backward Compatibility...")
    
    try:
        from models import ReportGeneration
        
        # Create test data
        test_df = pd.DataFrame({
            'field_intelligence_translated': ['Legacy test feedback'],
            'submission_datetime': ['2024-01-01'],
            'class': ['Legacy Issue'],
            'location_name': ['Legacy Location'],
            'form_type': ['CONSUMER_FEEDBACK']
        })
        
        # Test the old method still works
        prompt = ReportGeneration.create_generic_prompt_report(test_df)
        
        if prompt and len(prompt) > 500 and not prompt.startswith("Error:"):
            print("  ✅ Legacy interface still functional")
            return True
        else:
            print(f"  ❌ Legacy interface broken: {prompt[:100] if prompt else 'None'}...")
            return False
            
    except Exception as e:
        print(f"  ❌ Backward compatibility test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Report System Tests")
    print("=" * 50)
    
    tests = [
        test_xml_processing,
        test_template_manager,
        test_report_extractor,
        test_backward_compatibility
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_func.__name__} FAILED")
        except Exception as e:
            print(f"❌ {test_func.__name__} CRASHED: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The improved report system is working correctly.")
        print("\nKey improvements verified:")
        print("  ✅ No crashes during processing")
        print("  ✅ Graceful error handling")
        print("  ✅ Robust XML processing")
        print("  ✅ Multiple extraction strategies")
        print("  ✅ Backward compatibility maintained")
        return True
    else:
        print(f"⚠️  {total - passed} tests failed. Please check the error messages above.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {e}")
        traceback.print_exc()
        sys.exit(1) 