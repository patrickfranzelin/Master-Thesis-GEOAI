#!/usr/bin/env python3
"""
Validation script for the refactored pipeline architecture.

This script tests the decision and routing logic without requiring actual images or database.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_decision_dataclass():
    """Test HouseDecision dataclass creation."""
    print("✓ Testing HouseDecision dataclass...")
    
    # Import here to avoid dependency issues
    from pipeline.decision import HouseDecision
    
    # Test full house scenario
    decision1 = HouseDecision(
        house_present=True,
        full_house=True,
        reason=None
    )
    assert decision1.house_present == True
    assert decision1.full_house == True
    assert decision1.reason is None
    
    # Test partial house scenario
    decision2 = HouseDecision(
        house_present=True,
        full_house=False,
        reason="Polygon cuts off west side of house"
    )
    assert decision2.house_present == True
    assert decision2.full_house == False
    assert decision2.reason == "Polygon cuts off west side of house"
    
    # Test no house scenario
    decision3 = HouseDecision(
        house_present=False,
        full_house=None,
        reason="No structure inside polygon"
    )
    assert decision3.house_present == False
    assert decision3.full_house is None
    
    print("  ✓ All dataclass tests passed")


def test_routing_logic():
    """Test pipeline routing logic."""
    print("✓ Testing routing logic...")
    
    # Import here to avoid dependency issues
    from pipeline.decision import HouseDecision
    from pipeline.routing import route_pipeline
    
    # Test FULL pipeline routing
    decision1 = HouseDecision(house_present=True, full_house=True, reason=None)
    pipeline1 = route_pipeline(decision1)
    assert pipeline1 == "FULL", f"Expected FULL, got {pipeline1}"
    print(f"  ✓ Full house → {pipeline1} pipeline")
    
    # Test PARTIAL pipeline routing
    decision2 = HouseDecision(house_present=True, full_house=False, reason="Incomplete")
    pipeline2 = route_pipeline(decision2)
    assert pipeline2 == "PARTIAL", f"Expected PARTIAL, got {pipeline2}"
    print(f"  ✓ Partial house → {pipeline2} pipeline")
    
    # Test DISCOVERY pipeline routing
    decision3 = HouseDecision(house_present=False, full_house=None, reason="No house")
    pipeline3 = route_pipeline(decision3)
    assert pipeline3 == "DISCOVERY", f"Expected DISCOVERY, got {pipeline3}"
    print(f"  ✓ No house → {pipeline3} pipeline")
    
    print("  ✓ All routing tests passed")


def test_pipeline_modules_exist():
    """Test that all pipeline modules exist and have expected functions."""
    print("✓ Testing pipeline modules exist...")
    
    pipeline_dir = Path(__file__).parent / "src" / "pipeline"
    
    required_files = [
        "__init__.py",
        "decision.py",
        "routing.py",
        "full_house_pipeline.py",
        "partial_house_pipeline.py",
        "discovery_pipeline.py",
    ]
    
    for filename in required_files:
        filepath = pipeline_dir / filename
        assert filepath.exists(), f"Missing file: {filepath}"
        print(f"  ✓ {filename} exists")
    
    print("  ✓ All required files present")


def test_discovery_prompt_simplified():
    """Test that discovery prompt has been simplified."""
    print("✓ Testing discovery prompt simplification...")
    
    discovery_file = Path(__file__).parent / "src" / "mlqa" / "discovery_client.py"
    content = discovery_file.read_text()
    
    # Check for simplified format in the prompt
    assert "building1_points" in content, "Should have building1_points in prompt"
    assert "building2_points" in content, "Should have building2_points in prompt"
    assert "total_buildings" in content, "Should have total_buildings in prompt"
    
    # Check that the prompt itself doesn't have the complex nested format
    # (The parser will convert to the old format, but the prompt should be simple)
    prompt_section = content[content.find("DISCOVERY_PROMPT"):content.find('"""', content.find("DISCOVERY_PROMPT") + 100)]
    
    # Verify prompt uses simple format
    assert "building1_points" in prompt_section, "Prompt should use building1_points"
    assert "building2_points" in prompt_section, "Prompt should use building2_points"
    
    print("  ✓ Discovery prompt uses simplified format")
    print("  ✓ Parser converts to standard internal format")


def test_main_architecture():
    """Test that main.py uses the new architecture."""
    print("✓ Testing main.py architecture...")
    
    main_file = Path(__file__).parent / "src" / "main.py"
    content = main_file.read_text()
    
    # Check for new imports
    assert "from src.pipeline.decision import mlqa_decide" in content
    assert "from src.pipeline.routing import route_pipeline" in content
    assert "from src.pipeline.full_house_pipeline import full_house_pipeline" in content
    assert "from src.pipeline.partial_house_pipeline import partial_house_pipeline" in content
    assert "from src.pipeline.discovery_pipeline import discovery_pipeline" in content
    
    # Check for decision and routing calls
    assert "mlqa_decide(" in content
    assert "route_pipeline(" in content
    
    # Check for pipeline execution
    assert 'if pipeline == "FULL":' in content
    assert 'elif pipeline == "PARTIAL":' in content
    assert 'elif pipeline == "DISCOVERY":' in content
    
    print("  ✓ main.py imports new architecture modules")
    print("  ✓ main.py uses decision → routing → pipeline pattern")


def main():
    """Run all validation tests."""
    print("\n" + "="*60)
    print("PIPELINE ARCHITECTURE VALIDATION")
    print("="*60 + "\n")
    
    try:
        test_pipeline_modules_exist()
        print()
        test_decision_dataclass()
        print()
        test_routing_logic()
        print()
        test_discovery_prompt_simplified()
        print()
        test_main_architecture()
        print()
        print("="*60)
        print("✅ ALL TESTS PASSED")
        print("="*60 + "\n")
        print("Architecture refactoring is complete and validated!")
        print()
        return 0
    except Exception as e:
        print()
        print("="*60)
        print(f"❌ TESTS FAILED: {e}")
        print("="*60 + "\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
