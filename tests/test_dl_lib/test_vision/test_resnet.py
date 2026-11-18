# import unittest
#
# import torch
#
# from dl_lib.vision.resnet import ResidualBlock, ResNet
#
#
# class TestResidualBlockInit(unittest.TestCase):
#     def test_when_in_and_out_channels_match_then_has_no_shortcut_projection(self):
#         # Arrange
#         block = ResidualBlock(in_channels=16, out_channels=16, stride=1)
#
#         # Act / Assert
#         self.assertIsNone(block.shortcut_conv)
#         self.assertIsNone(block.shortcut_bn)
#
#     def test_when_out_channels_differs_from_in_channels_then_has_a_shortcut_projection(self):
#         # Arrange
#         block = ResidualBlock(in_channels=16, out_channels=32, stride=2)
#
#         # Act / Assert
#         self.assertIsInstance(block.shortcut_conv, torch.nn.Conv2d)
#         self.assertIsInstance(block.shortcut_bn, torch.nn.BatchNorm2d)
#
#     def test_when_conv1_is_applied_then_output_channels_and_spatial_size_match_stride(self):
#         # Arrange
#         block = ResidualBlock(in_channels=16, out_channels=32, stride=2)
#         x = torch.randn(2, 16, 8, 8)
#         expected = torch.Size([2, 32, 4, 4])
#
#         # Act
#         actual = block.conv1(x).shape
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#     def test_when_bn1_is_applied_to_conv1_output_then_shape_is_unchanged(self):
#         # Arrange
#         block = ResidualBlock(in_channels=16, out_channels=32, stride=2)
#         x = torch.randn(2, 16, 8, 8)
#         conv1_out = block.conv1(x)
#
#         # Act
#         actual = block.bn1(conv1_out).shape
#
#         # Assert
#         self.assertEqual(actual, conv1_out.shape)
#
#     def test_when_relu_is_applied_then_every_output_value_is_non_negative(self):
#         # Arrange
#         block = ResidualBlock(in_channels=16, out_channels=16, stride=1)
#         x = torch.randn(4, 16, 8, 8)
#
#         # Act
#         actual = block.relu(x)
#
#         # Assert
#         self.assertTrue((actual >= 0).all())
#
#     def test_when_conv2_is_applied_then_spatial_size_and_channels_stay_the_same(self):
#         # Arrange
#         block = ResidualBlock(in_channels=16, out_channels=32, stride=2)
#         x = torch.randn(2, 32, 4, 4)
#         expected = torch.Size([2, 32, 4, 4])
#
#         # Act
#         actual = block.conv2(x).shape
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#     def test_when_shortcut_conv_is_applied_then_output_matches_the_main_paths_shape(self):
#         # Arrange
#         block = ResidualBlock(in_channels=16, out_channels=32, stride=2)
#         x = torch.randn(2, 16, 8, 8)
#         expected = torch.Size([2, 32, 4, 4])
#
#         # Act
#         actual = block.shortcut_conv(x).shape
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#
# class TestResidualBlockLayers(unittest.TestCase):
#     def test_when_no_shortcut_then_returns_the_main_paths_layers_in_order(self):
#         # Arrange
#         block = ResidualBlock(in_channels=16, out_channels=16, stride=1)
#         expected = [block.conv1, block.bn1, block.relu, block.conv2, block.bn2]
#
#         # Act
#         actual = block._layers()
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#     def test_when_projecting_then_also_appends_the_shortcuts_layers(self):
#         # Arrange
#         block = ResidualBlock(in_channels=16, out_channels=32, stride=2)
#         expected = [
#             block.conv1,
#             block.bn1,
#             block.relu,
#             block.conv2,
#             block.bn2,
#             block.shortcut_conv,
#             block.shortcut_bn,
#         ]
#
#         # Act
#         actual = block._layers()
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#
# class TestResidualBlockForward(unittest.TestCase):
#     def test_when_in_and_out_channels_match_then_output_shape_equals_input_shape(self):
#         # Arrange
#         block = ResidualBlock(in_channels=16, out_channels=16, stride=1)
#         x = torch.randn(2, 16, 8, 8)
#         expected = x.shape
#
#         # Act
#         actual = block(x).shape
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#     def test_when_projecting_then_output_shape_matches_the_new_channels_and_spatial_size(self):
#         # Arrange
#         block = ResidualBlock(in_channels=16, out_channels=32, stride=2)
#         x = torch.randn(2, 16, 8, 8)
#         expected = torch.Size([2, 32, 4, 4])
#
#         # Act
#         actual = block(x).shape
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#
# class TestResidualBlockParameters(unittest.TestCase):
#     def test_when_no_shortcut_then_returns_parameters_from_the_main_path_only(self):
#         # Arrange
#         block = ResidualBlock(in_channels=16, out_channels=16, stride=1)
#         expected = (
#             len(list(block.conv1.parameters()))
#             + len(list(block.bn1.parameters()))
#             + len(list(block.conv2.parameters()))
#             + len(list(block.bn2.parameters()))
#         )
#
#         # Act
#         actual = len(block.parameters())
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#     def test_when_projecting_then_also_includes_the_shortcuts_parameters(self):
#         # Arrange
#         block = ResidualBlock(in_channels=16, out_channels=32, stride=2)
#         main_path_count = (
#             len(list(block.conv1.parameters()))
#             + len(list(block.bn1.parameters()))
#             + len(list(block.conv2.parameters()))
#             + len(list(block.bn2.parameters()))
#         )
#         expected = (
#             main_path_count
#             + len(list(block.shortcut_conv.parameters()))
#             + len(list(block.shortcut_bn.parameters()))
#         )
#
#         # Act
#         actual = len(block.parameters())
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#
# class TestResidualBlockTrain(unittest.TestCase):
#     def test_when_called_after_eval_then_sets_every_sublayer_to_training_mode(self):
#         # Arrange
#         block = ResidualBlock(in_channels=16, out_channels=32, stride=2)
#         block.eval()
#
#         # Act
#         block.train()
#
#         # Assert
#         self.assertTrue(block.training)
#         self.assertTrue(block.conv1.training)
#         self.assertTrue(block.bn1.training)
#         self.assertTrue(block.conv2.training)
#         self.assertTrue(block.bn2.training)
#         self.assertTrue(block.shortcut_conv.training)
#         self.assertTrue(block.shortcut_bn.training)
#
#     def test_when_called_then_returns_the_block_itself(self):
#         # Arrange
#         block = ResidualBlock(in_channels=16, out_channels=16, stride=1)
#
#         # Act
#         actual = block.train()
#
#         # Assert
#         self.assertIs(actual, block)
#
#
# class TestResidualBlockEval(unittest.TestCase):
#     def test_when_called_then_sets_every_sublayer_to_evaluation_mode(self):
#         # Arrange
#         block = ResidualBlock(in_channels=16, out_channels=32, stride=2)
#
#         # Act
#         block.eval()
#
#         # Assert
#         self.assertFalse(block.training)
#         self.assertFalse(block.conv1.training)
#         self.assertFalse(block.bn1.training)
#         self.assertFalse(block.conv2.training)
#         self.assertFalse(block.bn2.training)
#         self.assertFalse(block.shortcut_conv.training)
#         self.assertFalse(block.shortcut_bn.training)
#
#     def test_when_called_then_returns_the_block_itself(self):
#         # Arrange
#         block = ResidualBlock(in_channels=16, out_channels=16, stride=1)
#
#         # Act
#         actual = block.eval()
#
#         # Assert
#         self.assertIs(actual, block)
#
#
# class TestResidualBlockTo(unittest.TestCase):
#     def test_when_called_with_cpu_then_every_parameter_stays_on_cpu(self):
#         # Arrange
#         block = ResidualBlock(in_channels=16, out_channels=32, stride=2)
#         expected = torch.device("cpu")
#
#         # Act
#         block.to(expected)
#
#         # Assert
#         for param in block.parameters():
#             self.assertEqual(param.device, expected)
#
#     def test_when_called_then_returns_the_block_itself(self):
#         # Arrange
#         block = ResidualBlock(in_channels=16, out_channels=16, stride=1)
#
#         # Act
#         actual = block.to(torch.device("cpu"))
#
#         # Assert
#         self.assertIs(actual, block)
#
#
# class TestResNetInit(unittest.TestCase):
#     def test_when_depth_does_not_satisfy_six_n_plus_two_then_raises_value_error(self):
#         # Act / Assert
#         with self.assertRaises(ValueError):
#             ResNet(depth=21)
#
#     def test_when_global_avg_pool_is_true_then_head_starts_with_an_adaptive_avg_pool(self):
#         # Arrange
#         model = ResNet(depth=20, global_avg_pool=True)
#
#         # Act
#         actual = type(model.head[0])
#
#         # Assert
#         self.assertEqual(actual, torch.nn.AdaptiveAvgPool2d)
#
#     def test_when_global_avg_pool_is_false_then_head_has_far_more_parameters(self):
#         # Arrange
#         model_with_gap = ResNet(depth=20, global_avg_pool=True)
#         model_without_gap = ResNet(depth=20, global_avg_pool=False)
#
#         # Act
#         gap_head_params = sum(p.numel() for p in model_with_gap.head.parameters())
#         flatten_head_params = sum(p.numel() for p in model_without_gap.head.parameters())
#
#         # Assert
#         self.assertLess(gap_head_params, flatten_head_params)
#
#     def test_when_constructed_then_each_stage_has_the_expected_number_of_blocks(self):
#         # Arrange
#         model = ResNet(depth=20)
#         expected_blocks_per_stage = 3
#
#         # Act / Assert
#         self.assertEqual(len(model.stage1), expected_blocks_per_stage)
#         self.assertEqual(len(model.stage2), expected_blocks_per_stage)
#         self.assertEqual(len(model.stage3), expected_blocks_per_stage)
#
#     def test_when_constructed_then_defaults_to_training_mode(self):
#         # Arrange
#         model = ResNet(depth=20)
#
#         # Act
#         actual = model.training
#
#         # Assert
#         self.assertTrue(actual)
#
#
# class TestResNetMakeStage(unittest.TestCase):
#     def test_when_called_then_returns_the_requested_number_of_blocks(self):
#         # Arrange
#         expected = 3
#
#         # Act
#         actual = ResNet._make_stage(16, 32, num_blocks=3, stride=2)
#
#         # Assert
#         self.assertEqual(len(actual), expected)
#
#     def test_when_called_then_only_the_first_block_projects_and_the_rest_keep_shape(self):
#         # Arrange
#         blocks = ResNet._make_stage(16, 32, num_blocks=3, stride=2)
#
#         # Act / Assert
#         self.assertIsNotNone(blocks[0].shortcut_conv)
#         self.assertIsNone(blocks[1].shortcut_conv)
#         self.assertIsNone(blocks[2].shortcut_conv)
#
#
# class TestResNetStages(unittest.TestCase):
#     def test_when_called_then_returns_stage1_stage2_and_stage3_in_order(self):
#         # Arrange
#         model = ResNet(depth=20)
#         expected = [model.stage1, model.stage2, model.stage3]
#
#         # Act
#         actual = model._stages()
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#
# class TestResNetTo(unittest.TestCase):
#     def test_when_called_with_cpu_then_every_parameter_stays_on_cpu(self):
#         # Arrange
#         model = ResNet(depth=20)
#         expected = torch.device("cpu")
#
#         # Act
#         model.to(expected)
#
#         # Assert
#         for param in model.parameters():
#             self.assertEqual(param.device, expected)
#
#     def test_when_called_then_returns_the_model_itself(self):
#         # Arrange
#         model = ResNet(depth=20)
#
#         # Act
#         actual = model.to(torch.device("cpu"))
#
#         # Assert
#         self.assertIs(actual, model)
#
#
# class TestResNetForward(unittest.TestCase):
#     def test_when_called_then_output_shape_matches_batch_size_and_num_classes(self):
#         # Arrange
#         model = ResNet(depth=20, num_classes=10)
#         x = torch.randn(4, 3, 64, 64)
#         expected = torch.Size([4, 10])
#
#         # Act
#         actual = model(x).shape
#
#         # Assert
#         self.assertEqual(actual, expected)
#
#     def test_when_called_then_output_is_finite(self):
#         # Arrange
#         torch.manual_seed(0)
#         model = ResNet(depth=20)
#         x = torch.randn(4, 3, 64, 64)
#
#         # Act
#         actual = model(x)
#
#         # Assert
#         self.assertFalse(torch.isnan(actual).any())
#         self.assertTrue(torch.isfinite(actual).all())
#
#     def test_when_evaluating_then_two_forward_passes_of_the_same_input_are_identical(self):
#         # Arrange
#         torch.manual_seed(0)
#         model = ResNet(depth=20)
#         model.eval()
#         x = torch.randn(2, 3, 64, 64)
#
#         # Act
#         first = model(x)
#         second = model(x)
#
#         # Assert
#         self.assertTrue(torch.equal(first, second))
#
#     def test_when_backward_is_called_then_every_parameter_gets_a_finite_gradient(self):
#         # Arrange
#         torch.manual_seed(0)
#         model = ResNet(depth=20, num_classes=10)
#         x = torch.randn(2, 3, 64, 64)
#         y = torch.tensor([0, 1])
#         loss_fn = torch.nn.CrossEntropyLoss()
#
#         # Act
#         loss = loss_fn(model(x), y)
#         loss.backward()
#
#         # Assert
#         total_grad_magnitude = 0.0
#         for param in model.parameters():
#             self.assertIsNotNone(param.grad)
#             self.assertFalse(torch.isnan(param.grad).any())
#             total_grad_magnitude += param.grad.abs().sum().item()
#         self.assertGreater(total_grad_magnitude, 0.0)
#
#     def test_when_trained_for_a_few_steps_then_loss_decreases(self):
#         # Arrange
#         torch.manual_seed(0)
#         model = ResNet(depth=20, num_classes=2)
#         x = torch.randn(8, 3, 64, 64)
#         y = torch.randint(0, 2, (8,))
#         loss_fn = torch.nn.CrossEntropyLoss()
#         optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
#
#         initial_loss = loss_fn(model(x), y).item()
#
#         # Act
#         for _ in range(20):
#             optimizer.zero_grad()
#             loss = loss_fn(model(x), y)
#             loss.backward()
#             optimizer.step()
#         final_loss = loss_fn(model(x), y).item()
#
#         # Assert
#         self.assertFalse(torch.isnan(torch.tensor(final_loss)))
#         self.assertLess(final_loss, initial_loss)
#
#
# class TestResNetTrain(unittest.TestCase):
#     def test_when_called_after_eval_then_sets_stem_stages_and_head_to_training_mode(self):
#         # Arrange
#         model = ResNet(depth=20)
#         model.eval()
#
#         # Act
#         model.train()
#
#         # Assert
#         self.assertTrue(model.training)
#         self.assertTrue(model.stem.training)
#         self.assertTrue(model.stage1[0].training)
#         self.assertTrue(model.head.training)
#
#     def test_when_called_then_returns_the_model_itself(self):
#         # Arrange
#         model = ResNet(depth=20)
#
#         # Act
#         actual = model.train()
#
#         # Assert
#         self.assertIs(actual, model)
#
#
# class TestResNetEval(unittest.TestCase):
#     def test_when_called_then_sets_stem_stages_and_head_to_evaluation_mode(self):
#         # Arrange
#         model = ResNet(depth=20)
#
#         # Act
#         model.eval()
#
#         # Assert
#         self.assertFalse(model.training)
#         self.assertFalse(model.stem.training)
#         self.assertFalse(model.stage1[0].training)
#         self.assertFalse(model.head.training)
#
#     def test_when_called_then_returns_the_model_itself(self):
#         # Arrange
#         model = ResNet(depth=20)
#
#         # Act
#         actual = model.eval()
#
#         # Assert
#         self.assertIs(actual, model)
#
#
# class TestResNetParameters(unittest.TestCase):
#     def test_when_called_then_returns_every_parameter_from_stem_stages_and_head(self):
#         # Arrange
#         model = ResNet(depth=20)
#         expected = len(list(model.stem.parameters()))
#         for stage in (model.stage1, model.stage2, model.stage3):
#             for block in stage:
#                 expected += len(block.parameters())
#         expected += len(list(model.head.parameters()))
#
#         # Act
#         actual = len(model.parameters())
#
#         # Assert
#         self.assertEqual(actual, expected)
#         self.assertGreater(actual, 0)
