from .spot import SPOT


def run_spot(init_data, test_data, q=1e-2, level=0.6):
    spot = SPOT(q)
    spot.fit(init_data=init_data, data=test_data)
    spot.initialize(level=level)
    result = spot.run()
    return result
