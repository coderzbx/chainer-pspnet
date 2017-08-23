import matplotlib
matplotlib.use('Agg')

import tornado.ioloop
import tornado.web

from conf.config import ServerInfo
from modelPSPNet import ModelPSPNet


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world")


class PSPNetHandler(tornado.web.RequestHandler):
    def get(self):
        self.set_header('Access-Control-Allow-Origin', '*')
        model = self.get_argument("model", "Cityscapes")
        # image = read_image("test.jpg")
        image = "test.jpg"
        self.set_header('Content-type', 'image/png')
        with open(image, 'rb') as f:
            data = f.read()

        pspnet = ModelPSPNet(model)
        self.write(pspnet.do(image_data=data))
        self.finish()

    def post(self, *args, **kwargs):
        self.set_header('Access-Control-Allow-Origin', '*')
        model = self.get_argument("model", "Cityscapes")
        print(self.request.headers)

        content_type = self.request.headers['Content-Type']

        image = None
        if content_type.startswith("multipart/form-data"):
            files = self.request.files
            if len(files) > 0:
                valid = True
                for _, v in files.items():
                    file = v[0]
                    image = file['body']
            else:
                valid = False
        else:
            valid = True
            image = self.request.body

        if valid:
            pspnet = ModelPSPNet(model)
            pred_data = pspnet.do(image_data=image)

            self.set_header('Content-type', 'image/png')
            self.write(pred_data)
        else:
            self.set_header('Content-type', 'application/json')
            self.write('{"code": "1", "msg": "file is invalid"}')

        self.finish()


def make_app():
    server_info = ServerInfo()
    context = server_info.context
    uri = r"/{}".format(context)
    print(uri)
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/favicon.ico", MainHandler),
        (uri, PSPNetHandler)
    ])

if __name__ == "__main__":
    app = make_app()
    server_info = ServerInfo()
    port = server_info.port
    app.listen(port)
    tornado.ioloop.IOLoop.current().start()