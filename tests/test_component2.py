import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import cv2
import asyncio

from app.ml_models.component2.inference import run_pneumonia_inference, InvalidXRayError
from app.ml_models.component2.model import get_ood_shield_data, get_tb_shield_data

class TestComponent2Pneumonia(unittest.TestCase):
    def setUp(self):
        # Create a dummy image
        self.dummy_image = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)

    def test_invalid_image_raises_error(self):
        """A random noise image should fail the OOD shield check and raise InvalidXRayError."""
        print("Testing invalid image...")
        with self.assertRaises(InvalidXRayError):
            run_pneumonia_inference(self.dummy_image)
        print("Invalid image raised InvalidXRayError as expected.")

    @patch('app.ml_models.component2.inference.get_autoencoder')
    def test_valid_image_passes_shield(self, mock_get_autoencoder):
        """If the extracted features are equal to the lung centroid, it should pass the OOD shield."""
        print("Testing valid image (mocked)...")
        # Load centroid
        centroid, threshold = get_ood_shield_data()
        
        # Mock encoder to return a vector equal to centroid
        mock_encoder = MagicMock()
        mock_encoder.predict.return_value = np.expand_dims(centroid, axis=0)
        mock_get_autoencoder.return_value = (None, mock_encoder)

        # Mock the main pneumonia model prediction to avoid actual inference in this test
        with patch('app.ml_models.component2.inference.get_pneumonia_model') as mock_get_pneumonia_model:
            mock_model = MagicMock()
            # Predict returns probability, e.g., [[0.1]] (Normal)
            mock_model.predict.return_value = np.array([[0.1]])
            mock_get_pneumonia_model.return_value = mock_model
            
            diagnosis, confidence, severity, heatmap_base64, heatmap_sev = run_pneumonia_inference(self.dummy_image)
            
            # Assertions
            self.assertEqual(diagnosis, "NORMAL")
            self.assertAlmostEqual(confidence, 10.0)
            self.assertEqual(severity, "N/A (Normal)")
            self.assertIsNone(heatmap_base64)
            self.assertIn("affected_area_percent", heatmap_sev)
            self.assertIn("mean_intensity", heatmap_sev)
            
            # Verify encoder was called
            mock_encoder.predict.assert_called_once()
            
        print("Valid image passed OOD shield and ran inference successfully.")

    @patch('app.ml_models.component2.inference.get_autoencoder')
    def test_tb_shield_rejects_closer_to_tb_centroid(self, mock_get_autoencoder):
        """TB shield should reject embeddings at the TB centroid."""
        tb_centroid, _ = get_tb_shield_data()

        mock_encoder = MagicMock()
        mock_encoder.predict.return_value = np.expand_dims(tb_centroid, axis=0)
        mock_get_autoencoder.return_value = (None, mock_encoder)

        with self.assertRaises(InvalidXRayError):
            run_pneumonia_inference(self.dummy_image)

    @patch('app.api.v1.component2_pneumonia.cv2.imdecode')
    @patch('app.api.v1.component2_pneumonia.run_pneumonia_inference')
    def test_api_predict_invalid_image(self, mock_inference, mock_imdecode):
        """API should return 400 Bad Request with detail 'Input a valid pneumonia xray' when InvalidXRayError is raised."""
        print("Testing API predict invalid image...")
        from app.api.v1.component2_pneumonia import predict_pneumonia
        from fastapi import UploadFile, HTTPException
        import io
        
        mock_imdecode.return_value = self.dummy_image
        mock_inference.side_effect = InvalidXRayError("Input a valid pneumonia xray")
        
        # Create a mock file
        fake_file = UploadFile(filename="test.jpg", file=io.BytesIO(b"fake image data"))
        
        # Run the async handler directly
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(predict_pneumonia(patient_id="12345", file=fake_file))
        except HTTPException as e:
            self.assertEqual(e.status_code, 400)
            self.assertEqual(e.detail, "Input a valid pneumonia xray")
            print("API returned 400 Bad Request with correct detail message.")
        else:
            self.fail("HTTPException was not raised")
        finally:
            loop.close()

if __name__ == '__main__':
    unittest.main()
