# import unittest
#
# import torch
#
# from dl_lib.nn import init
#
#
# class TestCalculateFanInAndFanOut(unittest.TestCase):
#     def test_when_called_with_a_linear_weight_then_returns_its_in_and_out_features(self):
#         # Arrange
#         tensor = torch.empty(64, 32)
#         expected = (32, 64)
#
#         # Act
#         actual = init._calculate_fan_in_and_fan_out(tensor)
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#     def test_when_called_with_a_conv_weight_then_multiplies_by_the_kernel_size(self):
#         # Arrange
#         tensor = torch.empty(64, 32, 3, 3)
#         expected = (32 * 3 * 3, 64 * 3 * 3)
#
#         # Act
#         actual = init._calculate_fan_in_and_fan_out(tensor)
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#     def test_when_called_with_a_one_dimensional_tensor_then_raises_value_error(self):
#         # Arrange
#         tensor = torch.empty(10)
#
#         # Act / Assert
#         with self.assertRaises(ValueError):
#             init._calculate_fan_in_and_fan_out(tensor)
#
#
# class TestCalculateGain(unittest.TestCase):
#     def test_when_called_with_relu_then_returns_sqrt_two(self):
#         # Arrange
#         expected = 2.0 ** 0.5
#
#         # Act
#         actual = init.calculate_gain("relu")
#
#         # Assert
#         self.assertAlmostEqual(actual, expected)
#
#     def test_when_called_with_linear_then_returns_one(self):
#         # Arrange
#         expected = 1.0
#
#         # Act
#         actual = init.calculate_gain("linear")
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#     def test_when_called_with_unsupported_nonlinearity_then_raises_value_error(self):
#         # Act / Assert
#         with self.assertRaises(ValueError):
#             init.calculate_gain("not-a-real-nonlinearity")
#
#
# class TestKaimingNormal(unittest.TestCase):
#     def test_when_called_then_mutates_the_tensor_in_place_and_returns_it(self):
#         # Arrange
#         torch.manual_seed(0)
#         tensor = torch.empty(64, 32, 3, 3)
#
#         # Act
#         actual = init.kaiming_normal_(tensor, nonlinearity="relu")
#
#         # Assert
#         self.assertIs(actual, tensor)
#
#     def test_when_called_with_fan_in_then_std_matches_the_relu_formula(self):
#         # Arrange
#         torch.manual_seed(0)
#         tensor = torch.empty(1024, 256)
#         fan_in = 256
#         expected_std = (2.0 / fan_in) ** 0.5
#
#         # Act
#         init.kaiming_normal_(tensor, mode="fan_in", nonlinearity="relu")
#         actual_std = tensor.std().item()
#
#         # Assert
#         self.assertAlmostEqual(actual_std, expected_std, delta=0.05 * expected_std)
#
#
# class TestKaimingUniform(unittest.TestCase):
#     def test_when_called_then_mutates_the_tensor_in_place_and_returns_it(self):
#         # Arrange
#         torch.manual_seed(0)
#         tensor = torch.empty(64, 32, 3, 3)
#
#         # Act
#         actual = init.kaiming_uniform_(tensor, nonlinearity="relu")
#
#         # Assert
#         self.assertIs(actual, tensor)
#
#     def test_when_called_with_fan_in_then_every_value_is_within_the_relu_bound(self):
#         # Arrange
#         torch.manual_seed(0)
#         tensor = torch.empty(1024, 256)
#         fan_in = 256
#         std = (2.0 / fan_in) ** 0.5
#         expected_bound = (3.0 ** 0.5) * std
#
#         # Act
#         init.kaiming_uniform_(tensor, mode="fan_in", nonlinearity="relu")
#         actual_max_abs = tensor.abs().max().item()
#
#         # Assert
#         self.assertLessEqual(actual_max_abs, expected_bound + 1e-6)
#
#
# class TestXavierNormal(unittest.TestCase):
#     def test_when_called_then_mutates_the_tensor_in_place_and_returns_it(self):
#         # Arrange
#         torch.manual_seed(0)
#         tensor = torch.empty(256, 256)
#
#         # Act
#         actual = init.xavier_normal_(tensor)
#
#         # Assert
#         self.assertIs(actual, tensor)
#
#     def test_when_called_then_std_matches_the_fan_in_fan_out_average_formula(self):
#         # Arrange
#         torch.manual_seed(0)
#         tensor = torch.empty(1024, 256)
#         fan_in, fan_out = 256, 1024
#         expected_std = (2.0 / (fan_in + fan_out)) ** 0.5
#
#         # Act
#         init.xavier_normal_(tensor)
#         actual_std = tensor.std().item()
#
#         # Assert
#         self.assertAlmostEqual(actual_std, expected_std, delta=0.05 * expected_std)
#
#
# class TestXavierUniform(unittest.TestCase):
#     def test_when_called_then_mutates_the_tensor_in_place_and_returns_it(self):
#         # Arrange
#         torch.manual_seed(0)
#         tensor = torch.empty(256, 256)
#
#         # Act
#         actual = init.xavier_uniform_(tensor)
#
#         # Assert
#         self.assertIs(actual, tensor)
#
#     def test_when_called_then_every_value_is_within_the_fan_in_fan_out_bound(self):
#         # Arrange
#         torch.manual_seed(0)
#         tensor = torch.empty(1024, 256)
#         fan_in, fan_out = 256, 1024
#         std = (2.0 / (fan_in + fan_out)) ** 0.5
#         expected_bound = (3.0 ** 0.5) * std
#
#         # Act
#         init.xavier_uniform_(tensor)
#         actual_max_abs = tensor.abs().max().item()
#
#         # Assert
#         self.assertLessEqual(actual_max_abs, expected_bound + 1e-6)
