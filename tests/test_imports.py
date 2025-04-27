#!/usr/bin/env python
"""
Simple test script to verify import paths are working correctly
"""
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Basic imports
try:
    import env_module
    from env_module.flag_frenzy_env import FlagFrenzyEnv
    print("✅ Successfully imported env_module.flag_frenzy_env")
except Exception as e:
    print(f"❌ Error importing env_module.flag_frenzy_env: {e}")

try:
    import config
    from config.constants import ENV_CONFIG
    print("✅ Successfully imported config.constants")
except Exception as e:
    print(f"❌ Error importing config.constants: {e}")

try:
    import models
    from models.model import FlagFrenzyModel
    print("✅ Successfully imported models.model")
except Exception as e:
    print(f"❌ Error importing models.model: {e}")

try:
    import analysis
    from analysis.attribution import visualize_attributions
    print("✅ Successfully imported analysis.attribution")
except Exception as e:
    print(f"❌ Error importing analysis.attribution: {e}")

try:
    import tests
    from tests.env_tests.basic_tests import sim_test
    print("✅ Successfully imported tests.env_tests.basic_tests")
except Exception as e:
    print(f"❌ Error importing tests.env_tests.basic_tests: {e}")

print("\nImport test complete")