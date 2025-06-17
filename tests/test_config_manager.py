import pytest
import os
from query_data_predictor.config_manager import ConfigManager
from fixtures import sample_config
import tempfile

class TestConfigManager:
    """Test suite for the configuration manager."""
    
    @pytest.fixture
    def temp_config_file(self, sample_config):
        """Fixture to create a temporary config file."""
        import yaml
        
        with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False) as temp_file:
            yaml.dump(sample_config, temp_file)
            temp_path = temp_file.name
        
        yield temp_path
        os.unlink(temp_path)
    
    def test_default_config(self):
        """Test loading default configuration."""
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        # Check some default values
        assert config['discretization']['enabled'] is True
        assert config['discretization']['bins'] == 5
        assert config['recommendation']['method'] == 'association_rules'
    
    def test_load_config(self, temp_config_file, sample_config):
        """Test loading configuration from file."""
        config_manager = ConfigManager(temp_config_file)
        config = config_manager.get_config()
        
        # Check loaded values match sample config
        assert config['discretization']['bins'] == 3
        assert config['association_rules']['min_threshold'] == 0.5
    
    def test_get_section(self, sample_config):
        """Test getting a specific config section."""
        config_manager = ConfigManager()
        config_manager.config.update(sample_config)
        
        discretization = config_manager.get_section('discretization')
        assert discretization['method'] == 'equal_width'
        assert discretization['bins'] == 3
    
    def test_is_enabled(self, sample_config):
        """Test checking if a component is enabled."""
        config_manager = ConfigManager()
        config_manager.config.update(sample_config)
        
        assert config_manager.is_enabled('discretization') is True
        assert config_manager.is_enabled('nonexistent') is False