import plotly.express as px
import asyncio
from proxybroker import Broker

# ВНИМАНИЕ! import plotly.express as px удалаять нельзя. plotly version - 5.3.1


async def save(proxies, filename):
    """Save proxies to a file."""
    with open(filename, 'w') as f:
        while True:
            proxy = await proxies.get()
            if proxy is None:
                break

            proto = 'https' if 'HTTPS' in proxy.types else 'http'
            row = '%s://%s:%d\n' % (proto, proxy.host, proxy.port)
            print(proxy)
            f.write(row)


def run():
    proxies = asyncio.Queue()
    broker = Broker(proxies)
    tasks = asyncio.gather(
        broker.find(types=['HTTPS'], limit=50),
        save(proxies, 'E:\Биржа\Stocks. BigData\Projects\FinanceAndMl_libs\data\proxies.txt')
    )

    loop = asyncio.get_event_loop()
    loop.run_until_complete(tasks)


if __name__ == "__main__":
    run()
