class TrainingContext:
  def __init__(self):
    self._training: bool = True

  def train(self):
    self._training = True

  def eval(self):
    self._training = False

  def is_training(self) -> bool:
    return self._training
