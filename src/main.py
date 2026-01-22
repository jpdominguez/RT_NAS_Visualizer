
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyOKAERTool import main as okt
# import ok.ok as ok
# import okaertool as okt
from pyNAVIS import *
import time
import matplotlib.pyplot as plt

time_start = time.perf_counter()


# 64 Channels,  Stereo, 16 bits address, recorded using jAER
NUM_CHANNELS = 64
BIN_SIZE = 20000

# pyNAVIS settings
SETTINGS = MainSettings(num_channels=NUM_CHANNELS, mono_stereo=1, on_off_both=1, address_size=2, ts_tick=0.01,
                        bin_size=BIN_SIZE, reset_timestamp=True)


# device = ok.okCFrontPanel()                 # Número de Serie de la OpalKelly: 1529000BQK
# #device.OpenBySerial("1529000BQK")         # Si NO se descomenta, falla la conexión USB con la OpalKelly
# serial_number = device.GetSerialNumber()
# print('Número de Serie', serial_number)
# device_count = device.GetDeviceCount()
# print('device_count =', device_count)
# for idx in range(device_count):
#     print(f"Device[{idx}] Model: {device.GetDeviceListModel(idx)}")

BIT_FILE = bitfile_path = 'src/CPNAS_okaertool.bit'
SEQ_FILE = 'src/my_file_sequenced_B'
opal = okt.Okaertool(bit_file=BIT_FILE)
opal.init()
#opal.device.OpenBySerial("") #---ILA DEBUGGING---
# opal.select_command('idle')
# opal.select_inputs(inputs=['Port_A'])
#opal.bypass(inputs=['Port_A'])
#time.sleep(2.5)
#opal.select_command('bypass')

# opal.__select_inputs__(inputs=['Port_B'])
# opal.__select_command__('monitor')

port_selected = 'port_a'
if port_selected == 'port_a':
    p = 0
elif port_selected == 'port_b':
    p = 1
elif port_selected == 'port_c':
    p = 2

spikes = opal.monitor(inputs=[port_selected], duration=5, live=None)   #uffer_length=(1024*1024*1), events=200000000, file=SEQ_FILE)
#print('--------------------------')
#time_elapsed = (time.perf_counter() - time_start)
#print('Tiempo Final de Simulación', time_elapsed)
#print('--------------------------')

#opal.select_inputs(inputs=['Port_A'])
# opal.debug_file(SEQ_FILE = 'src/my_file_sequenced_A',buffer_length=(1024*64))
#opal.select_command('idle')
# opal.debug(buffer_length=(1024*64), num_transfers=16)
#opal.sequencer(SEQ_FILE) 
spikes[p].min_ts = min(spikes[p].timestamps)
spikes[p].max_ts = max(spikes[p].timestamps)
Functions.adapt_timestamps(spikes[p], settings=SETTINGS)
Plots.spikegram(spikes[p], settings=SETTINGS)
Plots.sonogram(spikes[p], settings=SETTINGS)
plt.show()








exit()




"""
Script para probar la funcionalidad de live monitoring de la clase Okaertool.
Se crea un objeto de la clase Okaertool, se inicia el live monitoring en un hilo aparte,
y se recogen y muestran los spikes capturados durante un periodo de tiempo.
Al final se detiene el live monitoring y se muestra el total de spikes capturados.
"""
import time
import sys
import os
import logging
import threading

# Agregar el directorio python_package al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyOKAERTool import main as okt

# Configurar el logger para mostrar mensajes en la consola
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('LiveMonitoringTest')

# Ruta al archivo .bit
bitfile_path = 'src/CPNAS_okaertool.bit'

# Validar la existencia del archivo .bit
if not os.path.exists(bitfile_path): 
    logger.error(f"El archivo .bit no existe en la ruta especificada: {bitfile_path}")
    sys.exit(1)

try:
    # Crear un objeto de la clase Okaertool
    okaer = okt.Okaertool(bit_file=bitfile_path)
    okaer.init()
    okaer.reset_board()

    # Lista de entradas disponibles: 'port_a', 'port_b', 'port_c'
    INPUTS = ['port_a', 'port_b', 'port_c']

    # Iniciar el live monitoring en un hilo aparte
    inputs = ['port_c']  # Reemplaza con los nombres de los puertos que deseas monitorizar
    okaer.live_monitor(inputs=inputs)

    # Esperar un tiempo para acumular spikes
    monitoring_duration = 10  # segundos
    MONITOR_INTERVAL = 0.01  # Intervalo en segundos
    logger.info(f'Live monitoring for {monitoring_duration} seconds...')

    # Recoger y mostrar los spikes capturados durante el periodo de tiempo
    start_time = time.time()
    while time.time() - start_time < monitoring_duration:
        spikes = okaer.get_live_spikes()
        if spikes is not None:
            for input_name in inputs:
                input_index = INPUTS.index(input_name)
                spike_list = spikes[input_index]
                if spike_list is not None:
                    logger.info(f"Spikes from {input_name}: {spike_list.addresses} at {spike_list.timestamps}")
        time.sleep(MONITOR_INTERVAL)

    # Detener el live monitoring
    okaer.live_monitor_stop()
    logger.info('Live monitoring stopped.')

    # Resumen de spikes capturados
    logger.info("Resumen de spikes capturados:")
    for input_name in inputs:
        input_index = INPUTS.index(input_name)
        spike_list = okaer.spikes[input_index]
        if spike_list is not None:
            logger.info(f"Total spikes from {input_name}: {len(spike_list.addresses)}")

except Exception as e:
    logger.error(f"Error durante la ejecución: {e}")
finally:
    del okaer
