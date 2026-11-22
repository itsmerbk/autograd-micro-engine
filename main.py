class Val:
    def __init__(self, data, children=()):
        self.data, self.grad, self._prev, self._back = data, 0.0, set(children), lambda: None
    def __add__(self, o):
        out = Val(self.data + o.data, (self, o))
        def _b(): self.grad += out.grad; o.grad += out.grad
        out._back = _b; return out
    def __mul__(self, o):
        out = Val(self.data * o.data, (self, o))
        def _b(): self.grad += o.data * out.grad; o.grad += self.data * out.grad
        out._back = _b; return out
    def backward(self):
        topo, vis = [], set()
        def dfs(v):
            if v not in vis:
                vis.add(v)
                for c in v._prev: dfs(c)
                topo.append(v)
        dfs(self); self.grad = 1.0
        for v in reversed(topo): v._back()

if __name__ == "__main__":
    a, b = Val(3.0), Val(4.0); c = a * b + a; c.backward()
    assert c.data == 15.0 and a.grad == 5.0 and b.grad == 3.0
    print("[PASS] 090_autograd_micro_engine")
