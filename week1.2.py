def dy(x):
    return 2*x
def y(x):
    try:
        return x*x
    except:
        return float('inf')
x = 0
eta = 0.1
epsilon = 1e-8
history_x=[x]
while True:
    gradient = dy(x)
    last_x = x
    x = x - eta * gradient
    history_x.append(x)
    if(abs(y(last_x)-y(x)<epsilon)):
        break
print(history_x)

