import matplotlib
matplotlib.use('Agg')

import tornado.ioloop
import tornado.web

import base64

from conf.config import ServerInfo
from modelPSPNet import ModelPSPNet

from datasets import cityscapes_labels


class PSPException(Exception):
    def __init__(self, error_code):
        self._code = int(error_code)

    def error_info(self):
        if self._code == 1:
            info = {"code": "1", "msg": "file is invalid"}
        elif self._code == 2:
            info = {"code": "2", "msg": "request is invalid"}
        else:
            info = {"code": "99", "msg": "unknown error"}

        return info


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Content-type', 'application/json')
        help_info = {
            'code': '0',
            'msg': 'service help',
            'context': 'deeplearning',
            'capabilities': [
                {
                    'name': 'image',
                    'method': 'POST',
                    'msg': 'get png data from server',
                    'example': '/deeplearning/image',
                    'request': {
                        'Content-type': 'multipart/form-data'
                    },
                    'response': {
                        'Content-type': 'image/png',
                    },
                    'error': {
                        'Content-type': 'application/json',
                    }
                },
                {
                    'name': 'information',
                    'method': 'GET',
                    'msg': 'get png data from server',
                    'example': '/deeplearning/information',
                    'request': {

                    },
                    'response': {
                        'Content-type': 'application/json',
                    },
                    'error': {
                        'Content-type': 'application/json',
                    }
                }
            ]
        }
        self.write(help_info)


class PSPNetHandler(tornado.web.RequestHandler):

    def get(self):
        self.set_header('Access-Control-Allow-Origin', '*')
        try:
            if self.request.uri != '/deeplearning/information' and self.request.uri != '/deeplearning':
                raise PSPException(2)

            clr_info = {}
            for l in cityscapes_labels:
                if not l.ignoreInEval:
                    clr_info[l.name] = l.color

            print(clr_info)
            self.set_header('Content-type', 'application/json')
            self.write(clr_info)

        except PSPException as e:
            self.set_header('Content-type', 'application/json')
            self.write(e.error_info())
        except Exception as err:
            self.set_header('Content-type', 'application/json')
            err_info = err.args[0]
            self.write('{"code": "99", "msg": {}}'.format(err_info))
        except:
            self.set_header('Content-type', 'application/json')
            self.write('{"code": "99", "msg": "unknown exception"}')

        self.finish()

    def post(self, *args, **kwargs):
        self.set_header('Access-Control-Allow-Origin', '*')
        model = self.get_argument("model", "Cityscapes")
        print(self.request.headers)

        try:
            if self.request.uri != '/deeplearning' and self.request.uri != '/deeplearning/image':
                raise PSPException(2)

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

            if not valid:
                raise PSPException(1)

            pspnet = ModelPSPNet(model)
            with open('upload.jpg', 'wb') as f:
                f.write(image)
            pred_data = pspnet.do(image_data=image)

            with open('upload.png', 'wb') as f1:
                f1.write(pred_data)

            self.set_header('Content-type', 'image/png')
            self.write(base64.b64encode(pred_data))

        except PSPException as e:
            self.set_header('Content-type', 'application/json')
            self.write(e.error_info())
        except Exception as err:
            self.set_header('Content-type', 'application/json')
            err_info = err.args[0]
            self.write('{"code": "99", "msg": {}}'.format(err_info))
        except:
            self.set_header('Content-type', 'application/json')
            self.write('{"code": "99", "msg": "unknown exception"}')

        self.finish()


def make_app():
    server_info = ServerInfo()
    context = server_info.context
    uri = r"/{}".format(context)
    uri_image = r"/{}/{}".format(context, 'image')
    uri_information = r"/{}/{}".format(context, 'information')
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/favicon.ico", MainHandler),
        (uri, PSPNetHandler),
        (uri_image, PSPNetHandler),
        (uri_information, PSPNetHandler)
    ])

if __name__ == "__main__":
    app = make_app()
    server_info = ServerInfo()
    port = server_info.port
    app.listen(port)
    tornado.ioloop.IOLoop.current().start()