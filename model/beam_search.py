import torch


class BeamSearch:
    def __init__(self,
                 end_index: int,
                 max_steps: int = 50,
                 beam_size: int = 10,
                 per_node_beam_size: int = None) -> None:
        self._end_index = end_index
        self.max_steps = max_steps
        self.beam_size = beam_size
        self.per_node_beam_size = per_node_beam_size or beam_size

    def search(self, start_predictions, start_state, step):
        batch_size = start_predictions.size()[0]
        predictions = []
        backpointers = []
        start_class_log_probabilities, state = step(
            start_predictions, start_state)

        num_classes = start_class_log_probabilities.size()[1]
        start_top_log_probabilities, start_predicted_classes = start_class_log_probabilities.topk(self.beam_size)
        if self.beam_size == 1 and (start_predicted_classes == self._end_index).all():
            print("Empty sequences predicted. You may want to "
                  "increase the beam size or ensure "
                  "your step function is working properly.")
            return start_predicted_classes.unsqueeze(-1), start_top_log_probabilities
        last_log_probabilities = start_top_log_probabilities
        predictions.append(start_predicted_classes)
        log_probs_after_end = start_class_log_probabilities.new_full(
            (batch_size * self.beam_size, num_classes),
            float("-inf")
        )
        log_probs_after_end[:, self._end_index] = 0.
        for key, state_tensor in state.items():
            _, *last_dims = state_tensor.size()
            state[key] = state_tensor.unsqueeze(1).expand(batch_size, self.beam_size, *last_dims).reshape(batch_size * self.beam_size, *last_dims)
        for timestep in range(self.max_steps - 1):
            last_predictions = predictions[-1].reshape(batch_size * self.beam_size)
            if (last_predictions == self._end_index).all():
                break
            class_log_probabilities, state = step(last_predictions, state)
            last_predictions_expanded = last_predictions.unsqueeze(-1).expand(
                batch_size * self.beam_size,
                num_classes
            )
            cleaned_log_probabilities = torch.where(
                last_predictions_expanded == self._end_index,
                log_probs_after_end,
                class_log_probabilities
            )
            top_log_probabilities, predicted_classes = cleaned_log_probabilities.topk(self.per_node_beam_size)
            expanded_last_log_probabilities = last_log_probabilities. unsqueeze(2).expand(batch_size, self.beam_size, self.per_node_beam_size).reshape(batch_size * self.beam_size, self.per_node_beam_size)
            summed_top_log_probabilities = top_log_probabilities + expanded_last_log_probabilities
            reshaped_summed = summed_top_log_probabilities.reshape(batch_size, self.beam_size * self.per_node_beam_size)
            reshaped_predicted_classes = predicted_classes.reshape(batch_size, self.beam_size * self.per_node_beam_size)
            restricted_beam_log_probs, restricted_beam_indices = reshaped_summed.topk(self.beam_size)
            restricted_predicted_classes = reshaped_predicted_classes.gather(1, restricted_beam_indices)
            predictions.append(restricted_predicted_classes)
            last_log_probabilities = restricted_beam_log_probs
            backpointer = restricted_beam_indices / self.per_node_beam_size
            backpointers.append(backpointer)
            for key, state_tensor in state.items():
                _, *last_dims = state_tensor.size()
                expanded_backpointer = backpointer.view(batch_size, self.beam_size, *([1] * len(last_dims))).expand(batch_size, self.beam_size, *last_dims)
                state[key] = state_tensor.reshape(batch_size, self.beam_size, *last_dims).gather(1, expanded_backpointer).reshape(batch_size * self.beam_size, *last_dims)
        if not torch.isfinite(last_log_probabilities).all():
            print("Infinite log probabilities encountered. "
                  "Some final sequences may not make sense. "
                  "This can happen when the beam size is "
                  "larger than the number of valid (non-zero "
                  "probability) transitions that the step function produces.")
        reconstructed_predictions = [predictions[-1].unsqueeze(2)]
        cur_backpointers = backpointers[-1]
        for timestep in range(len(predictions) - 2, 0, -1):
            cur_preds = predictions[timestep].gather(1, cur_backpointers).unsqueeze(2)
            reconstructed_predictions.append(cur_preds)
            cur_backpointers = backpointers[timestep - 1].gather(1, cur_backpointers)
        final_preds = predictions[0].gather(1, cur_backpointers).unsqueeze(2)
        reconstructed_predictions.append(final_preds)
        all_predictions = torch.cat(list(reversed(reconstructed_predictions)), 2)

        return all_predictions, last_log_probabilities