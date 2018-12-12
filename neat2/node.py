import Utils
import connectiongene

class Node:
    def __init__(self, num, activation):
        self.number = num
        self.input_sum = 0
        self.output_value = 0
        self.output_connections = []
        self.layer = 0
        self.activation = getattr(Utils, activation)

    def fire(self):
        if self.layer != 0:
            self.output_value = self.activation(self.input_sum)
        for next_connections in self.output_connections:
            if next_connections.enabled:
                next_connections.to_node.input_sum += next_connections.weight * self.output_value

    def is_connected(self, node):
        if self.layer == node.layer:
            return False
        elif node.layer < node.layer:
            return self in (x.to_node for x in node.output_connections)
        else:
            return self in (x.to_node for x in self.output_connections)

    def __eq__(self, other):
        assert isinstance(other, Node)
        return self.number == other.number
