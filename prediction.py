

class Prediction:
  def __init__(self, label, prob):
    self.label = label
    self.prob = prob

  def serialize(self):
    return {
      'label': self.label,
      'prob': float(self.prob)
    }