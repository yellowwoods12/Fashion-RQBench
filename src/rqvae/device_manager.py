"""Simple device management for training scripts."""

import logging

import torch


class DeviceManager:
    """Simple device management for training scripts."""

    def __init__(self, logger: logging.Logger):
        """Initialize device manager.

        Args:
            logger: Logger instance from the main script
        """
        self.logger = logger
        self.device = self._select_device()
        self._configure_device()

    def _select_device(self) -> str:
        """Select the best available device."""
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        self.logger.info(f"Using device: {device}")
        return device

    def _configure_device(self):
        """Apply device-specific configurations."""
        if self.device == "cuda":
            # Enable TF32 on Ampere GPUs for faster training
            torch.set_float32_matmul_precision("high")
            self.logger.info("Enabled TF32 precision for matrix multiplications")

    @property
    def is_cuda(self) -> bool:
        """Check if device is CUDA."""
        return self.device == "cuda"

    @property
    def is_mps(self) -> bool:
        """Check if device is MPS (Apple Silicon)."""
        return self.device == "mps"

    @property
    def supports_compile(self) -> bool:
        """Check if device supports torch.compile."""
        return self.device == "cuda"

    @property
    def supports_pin_memory(self) -> bool:
        """Check if device supports pinned memory for DataLoader."""
        return self.device == "cuda"
