
def write_board(writer, loss, avg_reward, q, q_target, q_logits, q_action_values, q_target_action_values, y, epsilon, episode):
    """
    Write the player's action on the board
    """

    # main logs
    writer.add_scalar("Basics/training_loss", loss, episode)
    writer.add_scalar("Basics/eval_set_reward", avg_reward, episode)
    writer.add_scalar("Basics/epsilon", epsilon, episode)

    # Q network raw output
    writer.add_scalar("QNetworkLogits/q_logits_min", q_logits.min(), episode)
    writer.add_scalar("QNetworkLogits/q_logits_max", q_logits.max(), episode)
    writer.add_scalar("QNetworkLogits/q_logits_abs_mean", q_logits.abs().mean(), episode)
    writer.add_scalar("QNetworkLogits/q_logits_mean", q_logits.mean(), episode)


    #  predicted and ground truth output components
    writer.add_scalar("OutputElements/q_action_values", q_action_values.abs().mean().item(), episode)
    writer.add_scalar("OutputElements/y", y.abs().mean().item(), episode)
    writer.add_scalar("OutputElements/q_target_action_values_min", q_target_action_values.min(), episode)
    writer.add_scalar("OutputElements/q_target_action_values_max", q_target_action_values.max(), episode)
    writer.add_scalar("OutputElements/q_target_action_values_abs_mean", q_target_action_values.abs().mean(), episode)
    writer.add_scalar("OutputElements/q_target_action_values_mean", q_target_action_values.mean(), episode)

    # Q network weights
    writer.add_scalar("QNetwork/conv_layer1_abs_mean", q.conv_layer1.abs().mean(), episode)
    writer.add_scalar("QNetwork/bias_layer1_abs_mean", q.bias_layer1.abs().mean(), episode)
    writer.add_scalar("QNetwork/conv_layer2_abs_mean", q.conv_layer2.abs().mean(), episode)
    writer.add_scalar("QNetwork/bias_layer2_abs_mean", q.bias_layer2.abs().mean(), episode)
    writer.add_scalar("QNetwork/ffn_layer_abs_mean", q.ffn_layer.abs().mean(), episode)
    writer.add_scalar("QNetwork/ffn_bias_abs_mean", q.ffn_bias.abs().mean(), episode)
    writer.add_scalar("QNetwork/output_layer_abs_mean", q.output_layer.abs().mean(), episode)
    writer.add_scalar("QNetwork/output_bias_abs_mean", q.output_bias.abs().mean(), episode)
    
    # Q target network weights
    writer.add_scalar("QTargetNetwork/conv_layer1_abs_mean", q_target.conv_layer1.abs().mean(), episode)
    writer.add_scalar("QTargetNetwork/bias_layer1_abs_mean", q_target.bias_layer1.abs().mean(), episode)
    writer.add_scalar("QTargetNetwork/conv_layer2_abs_mean", q_target.conv_layer2.abs().mean(), episode)
    writer.add_scalar("QTargetNetwork/bias_layer2_abs_mean", q_target.bias_layer2.abs().mean(), episode)
    writer.add_scalar("QTargetNetwork/ffn_layer_abs_mean", q_target.ffn_layer.abs().mean(), episode)
    writer.add_scalar("QTargetNetwork/ffn_bias_abs_mean", q_target.ffn_bias.abs().mean(), episode)
    writer.add_scalar("QTargetNetwork/output_layer_abs_mean", q_target.output_layer.abs().mean(), episode)
    writer.add_scalar("QTargetNetwork/output_bias_abs_mean", q_target.output_bias.abs().mean(), episode)
