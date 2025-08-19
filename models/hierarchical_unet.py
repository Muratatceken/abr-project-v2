"""
DEPRECATED: Hierarchical U-Net Architecture

This hierarchical U-Net + S4 implementation has been replaced by ABRTransformerGenerator.
Use the new ABRTransformerGenerator for ABR signal generation instead.

For migration:
    from models import ABRTransformerGenerator
    
    # Old way (deprecated):
    # model = OptimizedHierarchicalUNet(...)
    
    # New way:
    model = ABRTransformerGenerator(
        input_channels=1,
        static_dim=4,
        sequence_length=200,
        d_model=256,
        n_layers=6,
        n_heads=8
    )
"""

# Deprecation Error Classes
class DeprecatedModelError(RuntimeError):
    """Error raised when trying to use deprecated U-Net models."""
    pass


class OptimizedHierarchicalUNet:
    """
    DEPRECATED: Hierarchical U-Net is removed. Use ABRTransformerGenerator.
    """
    
    def __init__(self, *args, **kwargs):
        raise DeprecatedModelError(
            "Hierarchical U-Net is removed. Use ABRTransformerGenerator instead.\n"
            "Example:\n"
            "  from models import ABRTransformerGenerator\n"
            "  model = ABRTransformerGenerator(\n"
            "      input_channels=1, static_dim=4, sequence_length=200,\n"
            "      d_model=256, n_layers=6, n_heads=8\n"
            "  )"
        )


# Backward compatibility aliases - all raise deprecation errors
class HierarchicalUNet(OptimizedHierarchicalUNet):
    """Deprecated alias for OptimizedHierarchicalUNet."""
    pass


class EnhancedHierarchicalUNet(OptimizedHierarchicalUNet):
    """Deprecated alias for OptimizedHierarchicalUNet."""
    pass