# import unittest
#
# import torch
#
# from dl_lib.nn.linear import Linear
#
#
# class TestLinearInit(unittest.TestCase):
#     def test_when_constructed_then_weight_and_bias_match_torch_nn_linear_shapes(self):
#         # Given: in_features and out_features matching a torch.nn.Linear
#         in_features = 4
#         out_features = 3
#         reference = torch.nn.Linear(in_features, out_features)
#
#         # When: we construct our Linear
#         linear = Linear(in_features, out_features)
#
#         # Then: weight and bias are shaped the same way
#         self.assertEqual(tuple(linear.weight.shape), tuple(reference.weight.shape))
#         self.assertEqual(tuple(linear.bias.shape), tuple(reference.bias.shape))
#
#     def test_when_constructed_then_weight_and_bias_require_grad(self):
#         # Given: a Linear layer
#         linear = Linear(in_features=4, out_features=3)
#
#         # When: we check requires_grad on its parameters
#         # Then: both weight and bias are learnable
#         self.assertTrue(linear.weight.requires_grad)
#         self.assertTrue(linear.bias.requires_grad)
#
#
# class TestLinearForward(unittest.TestCase):
#     def test_when_given_an_unbatched_input_then_output_shape_matches_out_features(self):
#         # Given: a Linear layer and a single unbatched input vector
#         linear = Linear(in_features=4, out_features=3)
#         x = torch.randn(4)
#
#         # When: we run it forward
#         actual = linear(x)
#
#         # Then: the output is shaped (out_features,)
#         self.assertEqual(tuple(actual.shape), (3,))
#
#     def test_when_given_a_batched_input_then_output_shape_matches_batch_and_out_features(self):
#         # Given: a Linear layer and a batch of input vectors
#         linear = Linear(in_features=4, out_features=3)
#         x = torch.randn(5, 4)
#
#         # When: we run it forward
#         actual = linear(x)
#
#         # Then: the output is shaped (batch, out_features)
#         self.assertEqual(tuple(actual.shape), (5, 3))
#
#     def test_when_given_a_small_config_then_matches_torch_nn_linear(self):
#         # Given: our Linear and a reference torch.nn.Linear with the same weights
#         torch.manual_seed(0)
#         ours = Linear(in_features=4, out_features=3)
#         reference = torch.nn.Linear(4, 3)
#         assert reference.bias is not None
#         with torch.no_grad():
#             ours.weight.copy_(reference.weight)
#             ours.bias.copy_(reference.bias)
#         x = torch.randn(2, 4)
#
#         # When: we run both forward on the same input
#         actual = ours(x)
#         expected = reference(x)
#
#         # Then: the outputs match
#         self.assertTrue(torch.allclose(actual, expected, atol=1e-5))
#
#     def test_when_given_alexnet_classifier_configs_then_matches_torch_nn_linear(self):
#         # Given: our Linear and a reference torch.nn.Linear shaped like AlexNet's classifier
#         configs = [(256, 1024), (1024, 1024), (1024, 10)]
#         for in_features, out_features in configs:
#             torch.manual_seed(1)
#             ours = Linear(in_features=in_features, out_features=out_features)
#             reference = torch.nn.Linear(in_features, out_features)
#             assert reference.bias is not None
#             with torch.no_grad():
#                 ours.weight.copy_(reference.weight)
#                 ours.bias.copy_(reference.bias)
#             x = torch.randn(2, in_features)
#
#             # When: we run both forward on the same input
#             actual = ours(x)
#             expected = reference(x)
#
#             # Then: the outputs match
#             self.assertTrue(torch.allclose(actual, expected, atol=1e-5))
#
#
# class TestLinearParameters(unittest.TestCase):
#     def test_when_parameters_is_called_then_it_returns_weight_and_bias(self):
#         # Given: a Linear layer
#         linear = Linear(in_features=4, out_features=3)
#
#         # When: we ask for its parameters
#         actual_parameters = linear.parameters()
#
#         # Then: it returns exactly weight and bias, in that order
#         self.assertEqual(len(actual_parameters), 2)
#         self.assertIs(actual_parameters[0], linear.weight)
#         self.assertIs(actual_parameters[1], linear.bias)
#
#     def test_when_called_then_returns_a_list_equal_to_weight_and_bias(self):
#         # Given: a Linear layer
#         linear = Linear(in_features=4, out_features=3)
#         expected = [linear.weight, linear.bias]
#
#         # When: we ask for its parameters
#         actual = linear.parameters()
#
#         # Then: the list equals weight and bias, in order
#         self.assertEqual(actual, expected)
#
#
# class TestLinearTo(unittest.TestCase):
#     def test_when_moved_to_cpu_then_it_returns_self(self):
#         # Given: a Linear layer
#         linear = Linear(in_features=4, out_features=3)
#
#         # When: we move it to the cpu device
#         actual = linear.to(torch.device("cpu"))
#
#         # Then: it returns the same instance
#         self.assertIs(actual, linear)
#
#     def test_when_moved_to_cpu_then_weight_and_bias_keep_their_shape_and_requires_grad(self):
#         # Given: a Linear layer
#         linear = Linear(in_features=4, out_features=3)
#         expected_weight_shape = tuple(linear.weight.shape)
#         expected_bias_shape = tuple(linear.bias.shape)
#
#         # When: we move it to the cpu device
#         linear.to(torch.device("cpu"))
#
#         # Then: weight and bias keep their shape and stay learnable
#         self.assertEqual(tuple(linear.weight.shape), expected_weight_shape)
#         self.assertEqual(tuple(linear.bias.shape), expected_bias_shape)
#         self.assertTrue(linear.weight.requires_grad)
#         self.assertTrue(linear.bias.requires_grad)
