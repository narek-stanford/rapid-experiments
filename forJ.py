# coding: utf-8
from py4j import java_collections
from py4j import java_gateway


gtwyCl = java_gateway.GatewayClient()
java_collections.JavaArray("foo", gtwyCl)
# AttributeError: 'NoneType' object has no attribute 'auto_field'

java_gateway.JavaGateway(self, 
	gateway_client=None, 
	auto_field=False, 
	python_proxy_port=25334, 
	start_callback_server=False, 
	auto_convert=False, 
	eager_load=False, 
	gateway_parameters=None, 
	callback_server_parameters=None, 
	python_server_entry_point=None)

