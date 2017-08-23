class ServerInfo:
    def __init__(self):
        self._port = 9008
        self._context = "deeplearning"
        self._serviceId = "pspnet"

    @property
    def port(self):
        return self._port

    @port.setter
    def port(self, port):
        self._port = port

    @property
    def context(self):
        return self._context

    @context.setter
    def context(self, context):
        self._context = context

    @property
    def service_id(self):
        return self._serviceId

    @service_id.setter
    def service_id(self, service_id):
        self._serviceId = service_id
