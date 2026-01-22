"""
Script para capturar eventos durante 10 segundos y luego visualizar su temporalidad.
Genera múltiples plots para analizar la distribución temporal de los eventos.
"""
import time
import sys
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Agregar el directorio python_package al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyOKAERTool import main as okt

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('EventCapturePlot')

# ============== CONFIGURACIÓN ==============
BITFILE_PATH = 'src/CPNAS_okaertool.bit'
SELECTED_INPUTS = ['port_c']  # Puertos a monitorizar
INPUTS = ['port_a', 'port_b', 'port_c']
MONITORING_DURATION = 2  # Duración de captura en segundos
TIMESTAMP_UNIT = 10e-9  # 10 nanosegundos por tick del FPGA
MAX_ADDRESSES = 256  # Número máximo de canales

# Buffer pre-asignado para captura rápida
# Con ~350K eventos/segundo × 10s = 3.5M eventos necesarios
BUFFER_SIZE = 5_000_000  # Capacidad para 5M eventos (margen de seguridad)
all_timestamps = np.zeros(BUFFER_SIZE, dtype=np.float64)
all_addresses = np.zeros(BUFFER_SIZE, dtype=np.uint16)
write_idx = 0

# ============== FUNCIONES ==============

def capture_events(okaer, duration):
    """
    Captura eventos del FPGA durante 'duration' segundos.
    Considera que:
    - Timestamps del FPGA tienen tick de 10ns (TIMESTAMP_UNIT)
    - Cada batch tiene timestamps relativos al primer evento del batch
    - Los timestamps del FPGA son circulares y no progresivos entre batches
    Retorna arrays numpy con timestamps (absolutos desde inicio) y addresses.
    """
    global write_idx
    
    logger.info(f'Iniciando captura de eventos por {duration} segundos...')
    logger.info(f'Por favor espera {duration} segundos completos...')
    start_time = time.time()
    batch_count = 0
    last_log_time = start_time
    events_count_last_log = 0
    
    # DEBUG: Contadores para diagnóstico
    polls_with_data = 0
    polls_without_data = 0
    total_polls = 0
    debug_batch_num = 0  # Contador independiente para debug
    
    # IMPORTANTE: Capturar durante TODOS los 'duration' segundos
    while True:
        current_time = time.time()
        elapsed = current_time - start_time
        total_polls += 1
        
        # Condición de salida: SOLO cuando se cumplan los 'duration' segundos
        if elapsed >= duration:
            logger.info(f'✓ Completados {duration} segundos de captura')
            break
        
        spikes = okaer.get_live_spikes()
        has_data_this_poll = False
        
        if spikes is not None:
            for input_name in SELECTED_INPUTS:
                input_index = INPUTS.index(input_name)
                spike_list = spikes[input_index]
                
                if (spike_list is not None and 
                    hasattr(spike_list, 'addresses') and 
                    hasattr(spike_list, 'timestamps')):
                    
                    raw_addresses = spike_list.addresses
                    raw_timestamps = spike_list.timestamps
                    
                    # DEBUG: Verificar si realmente hay datos
                    addr_len = len(raw_addresses) if raw_addresses is not None else 0
                    
                    if raw_addresses is not None and len(raw_addresses) > 0:
                        has_data_this_poll = True
                        
                        # Convertir a arrays numpy
                        if not isinstance(raw_addresses, np.ndarray):
                            raw_addresses = np.array(raw_addresses, dtype=np.uint16)
                        if not isinstance(raw_timestamps, np.ndarray):
                            raw_timestamps = np.array(raw_timestamps, dtype=np.float64)
                        
                        # DEBUG: Mostrar rango de timestamps RAW del FPGA
                        if debug_batch_num < 3 or (debug_batch_num >= 40 and debug_batch_num < 43):
                            fpga_min = raw_timestamps.min()
                            fpga_max = raw_timestamps.max()
                            logger.info(f"  DEBUG batch {debug_batch_num} @ {elapsed:.3f}s: {len(raw_addresses)} eventos | "
                                      f"FPGA raw: [{fpga_min:.0f}, {fpga_max:.0f}]")
                        
                        debug_batch_num += 1
                        
                        # ESTRATEGIA CORREGIDA:
                        # Los timestamps del FPGA son CIRCULARES y se repiten
                        # NO podemos usarlos para determinar el tiempo absoluto
                        # En su lugar, distribuimos los eventos del batch en el momento de captura
                        
                        # Convertir timestamps FPGA a segundos para obtener distribución relativa
                        fpga_timestamps_seconds = raw_timestamps * TIMESTAMP_UNIT
                        
                        if len(fpga_timestamps_seconds) > 0:
                            # Obtener distribución temporal DENTRO del batch (preserva orden temporal)
                            first_timestamp = fpga_timestamps_seconds[0]
                            relative_offsets = fpga_timestamps_seconds - first_timestamp
                            
                            # CLAVE: Usar elapsed como base temporal
                            # Los offsets relativos son pequeños (microsegundos dentro del batch)
                            absolute_timestamps = elapsed + relative_offsets
                        else:
                            absolute_timestamps = np.array([], dtype=np.float64)
                        
                        # Filtrar direcciones válidas
                        valid_mask = (raw_addresses >= 0) & (raw_addresses < MAX_ADDRESSES)
                        valid_addresses = raw_addresses[valid_mask]
                        valid_timestamps = absolute_timestamps[valid_mask]
                        
                        n_valid = len(valid_addresses)
                        n_total = len(raw_addresses)
                        
                        # DEBUG: Mostrar filtrado de direcciones
                        if debug_batch_num <= 5 or (debug_batch_num >= 40 and debug_batch_num < 43):
                            logger.info(f"  → FILTRO: {n_total} total, {n_valid} válidos | "
                                      f"addr range: [{raw_addresses.min()}, {raw_addresses.max()}]")
                        
                        # Escribir en buffer
                        if n_valid > 0:
                            if write_idx + n_valid < BUFFER_SIZE:
                                all_timestamps[write_idx:write_idx+n_valid] = valid_timestamps
                                all_addresses[write_idx:write_idx+n_valid] = valid_addresses
                                write_idx += n_valid
                                batch_count += 1
                            else:
                                logger.warning(f"⚠ Buffer lleno! No se pueden añadir {n_valid} eventos más")
        
        # Contadores de diagnóstico
        if has_data_this_poll:
            polls_with_data += 1
        else:
            polls_without_data += 1
        
        # Log de progreso cada segundo
        if current_time - last_log_time >= 1.0:
            events_this_second = write_idx - events_count_last_log
            logger.info(f"⏱ {elapsed:.1f}s / {duration}s | {write_idx:,} eventos totales "
                       f"({events_this_second:,} eventos/s) | Polls: {polls_with_data} con datos, {polls_without_data} vacíos")
            last_log_time = current_time
            events_count_last_log = write_idx
            polls_with_data = 0
            polls_without_data = 0
        
        time.sleep(0.001)  # 1ms entre polls
    
    logger.info(f'Captura completada. Total eventos: {write_idx:,} | Batches: {batch_count} | Total polls: {total_polls}')
    
    # DEBUG: Analizar rango de timestamps capturados
    if write_idx > 0:
        ts_min = all_timestamps[:write_idx].min()
        ts_max = all_timestamps[:write_idx].max()
        logger.info(f'DEBUG - Rango de timestamps: {ts_min:.6f}s a {ts_max:.6f}s (span: {ts_max-ts_min:.6f}s)')
    else:
        logger.warning('⚠ No se capturaron eventos durante los {duration} segundos')
    
    # Retornar solo los datos válidos
    return all_timestamps[:write_idx].copy(), all_addresses[:write_idx].copy()


def plot_temporal_analysis(timestamps, addresses):
    """
    Genera múltiples plots para analizar la temporalidad de los eventos.
    """
    if len(timestamps) == 0:
        logger.warning("No hay eventos para graficar")
        return
    
    # Determinar el rango temporal COMPLETO de la captura
    time_span = timestamps.max() if len(timestamps) > 0 else MONITORING_DURATION
    
    # Crear figura con múltiples subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # ======== 1. RASTER PLOT (Spike raster) ========
    ax1 = fig.add_subplot(gs[0, :])
    ax1.scatter(timestamps, addresses, s=1, alpha=0.5, c='blue', marker='|')
    ax1.set_xlabel('Tiempo (s)', fontsize=12)
    ax1.set_ylabel('Canal (dirección)', fontsize=12)
    ax1.set_title(f'Raster Plot - Distribución espaciotemporal de eventos ({len(timestamps):,} eventos)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlim(0, MONITORING_DURATION)  # Mostrar TODA la ventana de captura
    ax1.set_ylim(0, MAX_ADDRESSES)
    ax1.grid(True, alpha=0.3)
    
    # Añadir línea vertical para marcar el final de eventos
    if time_span < MONITORING_DURATION:
        ax1.axvline(time_span, color='red', linestyle='--', alpha=0.5, linewidth=2,
                   label=f'Fin de eventos ({time_span:.2f}s)')
        ax1.legend(loc='upper right')
    
    # ======== 2. HISTOGRAMA TEMPORAL (Event rate over time) ========
    ax2 = fig.add_subplot(gs[1, 0])
    # Usar MONITORING_DURATION para los bins, no solo el rango con eventos
    n_bins_time = max(50, int(MONITORING_DURATION * 10))  # 10 bins por segundo
    counts_time, bins_time, _ = ax2.hist(timestamps, bins=n_bins_time, 
                                          range=(0, MONITORING_DURATION),
                                          color='green', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Tiempo (s)', fontsize=11)
    ax2.set_ylabel('Eventos por bin', fontsize=11)
    ax2.set_title('Tasa de eventos en el tiempo', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, MONITORING_DURATION)  # Mostrar toda la ventana
    ax2.grid(True, alpha=0.3)
    
    # Calcular y mostrar estadísticas de tasa
    bin_width = bins_time[1] - bins_time[0]
    mean_rate = counts_time.mean() / bin_width if bin_width > 0 else 0
    max_rate = counts_time.max() / bin_width if bin_width > 0 else 0
    ax2.axhline(counts_time.mean(), color='red', linestyle='--', 
                label=f'Media: {mean_rate:.1f} eventos/s')
    ax2.legend()
    
    # ======== 3. HISTOGRAMA DE CANALES (Channel activity) ========
    ax3 = fig.add_subplot(gs[1, 1])
    channel_counts = np.bincount(addresses.astype(int), minlength=MAX_ADDRESSES)
    active_channels = np.where(channel_counts > 0)[0]
    
    ax3.bar(range(MAX_ADDRESSES), channel_counts, width=1.0, 
            color='purple', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Canal (dirección)', fontsize=11)
    ax3.set_ylabel('Total de eventos', fontsize=11)
    ax3.set_title(f'Distribución por canal ({len(active_channels)} canales activos)', 
                  fontsize=12, fontweight='bold')
    ax3.set_xlim(0, MAX_ADDRESSES)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ======== 4. ACTIVIDAD ACUMULADA EN EL TIEMPO ========
    ax4 = fig.add_subplot(gs[2, 0])
    sorted_times = np.sort(timestamps)
    cumulative_events = np.arange(1, len(sorted_times) + 1)
    ax4.plot(sorted_times, cumulative_events, color='orange', linewidth=2)
    ax4.set_xlabel('Tiempo (s)', fontsize=11)
    ax4.set_ylabel('Eventos acumulados', fontsize=11)
    ax4.set_title('Eventos acumulados en el tiempo', fontsize=12, fontweight='bold')
    ax4.set_xlim(0, MONITORING_DURATION)  # Mostrar toda la ventana
    ax4.grid(True, alpha=0.3)
    
    # Añadir línea de referencia de tasa constante
    if len(timestamps) > 0:
        expected_constant = np.linspace(0, len(timestamps), 100)
        time_constant = np.linspace(0, MONITORING_DURATION, 100)
        ax4.plot(time_constant, expected_constant, 'r--', alpha=0.5, 
                label='Tasa constante ideal')
        ax4.legend()
    
    # ======== 5. HEATMAP TEMPORAL (Actividad por canal en el tiempo) ========
    ax5 = fig.add_subplot(gs[2, 1])
    
    # Crear bins temporales y de canales
    n_time_bins = min(100, max(50, int(MONITORING_DURATION * 10)))  # 10 bins/segundo
    n_channel_bins = min(64, MAX_ADDRESSES)  # Reducir resolución de canales para visualización
    
    time_bins = np.linspace(0, MONITORING_DURATION, n_time_bins + 1)
    channel_bins = np.linspace(0, MAX_ADDRESSES, n_channel_bins + 1)
    
    # Crear histograma 2D
    heatmap, xedges, yedges = np.histogram2d(timestamps, addresses, 
                                              bins=[time_bins, channel_bins])
    
    # Plotear heatmap
    im = ax5.imshow(heatmap.T, aspect='auto', origin='lower', 
                    cmap='hot', interpolation='nearest',
                    extent=[0, MONITORING_DURATION, 0, MAX_ADDRESSES])
    ax5.set_xlabel('Tiempo (s)', fontsize=11)
    ax5.set_ylabel('Canal (dirección)', fontsize=11)
    ax5.set_title('Heatmap: Actividad espaciotemporal', fontsize=12, fontweight='bold')
    
    # Añadir colorbar
    cbar = plt.colorbar(im, ax=ax5)
    cbar.set_label('Eventos por bin', fontsize=10)
    
    # ======== TÍTULO GENERAL CON ESTADÍSTICAS ========
    total_duration = timestamps.max() if len(timestamps) > 0 else 0
    avg_rate = len(timestamps) / total_duration if total_duration > 0 else 0
    
    fig.suptitle(f'Análisis Temporal de Eventos - {len(timestamps):,} eventos en {total_duration:.2f}s de {MONITORING_DURATION}s captura '
                 f'(~{avg_rate:.1f} eventos/s durante actividad)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    logger.info("="*70)
    logger.info("ESTADÍSTICAS DE CAPTURA")
    logger.info("="*70)
    logger.info(f"Total de eventos: {len(timestamps):,}")
    logger.info(f"Duración de eventos: {total_duration:.3f} segundos (de {MONITORING_DURATION}s captura)")
    logger.info(f"Tasa promedio durante actividad: {avg_rate:.1f} eventos/segundo")
    logger.info(f"Tasa máxima instantánea: {max_rate:.1f} eventos/segundo")
    logger.info(f"Canales activos: {len(active_channels)}/{MAX_ADDRESSES}")
    
    if total_duration < MONITORING_DURATION:
        silence_duration = MONITORING_DURATION - total_duration
        logger.info(f"⚠ Periodo de silencio: {silence_duration:.1f}s ({silence_duration/MONITORING_DURATION*100:.1f}% del tiempo)")
    
    if len(active_channels) > 0:
        most_active_channel = channel_counts.argmax()
        logger.info(f"Canal más activo: {most_active_channel} ({channel_counts[most_active_channel]:,} eventos)")
        logger.info(f"Promedio por canal activo: {len(timestamps)/len(active_channels):.1f} eventos")
    
    logger.info("="*70)
    
    plt.show()


# ============== MAIN ==============

def main():
    """Función principal"""
    
    # Validar archivo .bit
    if not os.path.exists(BITFILE_PATH):
        logger.error(f"El archivo .bit no existe: {BITFILE_PATH}")
        sys.exit(1)
    
    try:
        # Inicializar FPGA
        logger.info("Inicializando FPGA...")
        okaer = okt.Okaertool(bit_file=BITFILE_PATH)
        okaer.init()
        okaer.reset_board()
        logger.info("✓ FPGA inicializado correctamente")
        
        # Iniciar live monitoring
        logger.info(f"Iniciando monitorización en {SELECTED_INPUTS}...")
        okaer.live_monitor(inputs=SELECTED_INPUTS)
        logger.info("✓ Live monitoring activo")
        
        print("\n" + "="*70)
        print(f"CAPTURANDO EVENTOS DURANTE {MONITORING_DURATION} SEGUNDOS...")
        print("="*70 + "\n")
        
        # Capturar eventos
        timestamps, addresses = capture_events(okaer, MONITORING_DURATION)
        
        # Detener live monitoring
        okaer.live_monitor_stop()
        logger.info("✓ Live monitoring detenido")
        
        # Generar visualizaciones
        print("\n" + "="*70)
        print("GENERANDO GRÁFICOS...")
        print("="*70 + "\n")
        
        plot_temporal_analysis(timestamps, addresses)
        
    except Exception as e:
        logger.error(f"Error durante la ejecución: {e}", exc_info=True)
        sys.exit(1)
    
    finally:
        try:
            del okaer
            logger.info("✓ Recursos liberados")
        except:
            pass


if __name__ == '__main__':
    main()
