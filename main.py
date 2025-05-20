# main.py -- Framework base per ricostruzione Feldkamp 3D, artefatti e visualizzazione 3D

import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from metrics import PSNR, rel_err, SSIM
import astra

# Moduli di SparseCT per generazione e visualizzazione volumi
from SparseCT import data as sparse_data
from SparseCT import visualization as sparse_vis

# ====== 1. CREAZIONE O CARICAMENTO DATASET SINTETICO 3D ======
SYN_DATA_PATH = "./images/ellipsoid_volumes.npy"
SAVE_SYN_DATA = True
N_VOLS = 25         # Come da specifiche, 25-30 volumi
VOL_SHAPE = (256, 256, 50)

if not os.path.exists(SYN_DATA_PATH):
    print("[INFO] Generazione volume sintetico 3D...")
    vols = sparse_data.get_data((N_VOLS,)+VOL_SHAPE)
    if SAVE_SYN_DATA:
        np.save(SYN_DATA_PATH, vols)
else:
    vols = np.load(SYN_DATA_PATH)
    print(f"[INFO] Dataset sintetico caricato: {vols.shape}")

# ====== 2. ESPERIMENTI: angoli limitati, con/senza rumore ======
# GEOMETRIE richieste (in gradi):
angle_tests = {
    "range_15": np.linspace(-15, 15, 11),
    "range_8.5": np.linspace(-8.5, 8.5, 11)
}
noise_levels = [0.0, 0.01]

results = []

for range_name, angles_deg in angle_tests.items():
    for nl in noise_levels:
        this_exp_metrics = []
        print(f"=== ESPERIMENTO: {range_name}, noise={nl} ===")
        for i in range(N_VOLS):   # Puoi provarlo su tutto il test set (o solo 1 per debug)
            vol = vols[i]
            # ASTRA vuole shape (Z, Y, X)
            vol_astra = np.copy(vol)  # già in (256,256,50). Eventualmente trasporre qui se necessario.

            # Geometry config -- detector semplificato, params esemplificativi
            n_proj = len(angles_deg)
            angles = np.deg2rad(angles_deg)
            det_row, det_col = vol.shape[1], vol.shape[2]
            vol_geom = astra.create_vol_geom(vol.shape[1], vol.shape[2], vol.shape[0])
            proj_geom = astra.create_proj_geom('cone', 1.0, 1.0, det_row, det_col, angles, 1000, 1500)

            # ID allocazione
            vol_id = astra.data3d.create('-vol', vol_geom, vol_astra.astype(np.float32))
            proj_id = astra.data3d.create('-proj3d', proj_geom)

            # FORWARD: volume -> proiezioni
            cfg_fp = astra.creators.astra_dict('FP3D_CUDA')
            cfg_fp['ProjectionDataId'] = proj_id
            cfg_fp['VolumeDataId'] = vol_id
            alg_fp_id = astra.algorithm.create(cfg_fp)
            astra.algorithm.run(alg_fp_id)
            projs = astra.data3d.get(proj_id)
            # Add noise
            if nl > 0.0:
                noise = np.random.normal(0, 1, projs.shape).astype(np.float32)
                noise = noise / np.linalg.norm(noise.flatten(), 2)
                projs = projs + noise * nl * np.linalg.norm(projs.flatten(), 2)
                projs[projs < 0] = 0
            # Salva alcune slice di proiezione (debug/analisi)
            if i == 0:
                tifffile.imwrite(f'images/Projs_{range_name}_nl{nl}_id{i}.tiff', projs.astype(np.float32))

            # RICOSTRUZIONE Feldkamp
            rec_id = astra.data3d.create('-vol', vol_geom)
            cfg_fdk = astra.creators.astra_dict('FDK_CUDA')
            cfg_fdk['ProjectionDataId'] = proj_id
            cfg_fdk['ReconstructionDataId'] = rec_id
            alg_fdk_id = astra.algorithm.create(cfg_fdk)
            astra.algorithm.run(alg_fdk_id)
            reco = astra.data3d.get(rec_id)
            tifffile.imwrite(f'images/Reco_FDK_{range_name}_nl{nl}_id{i}.tiff', reco.astype(np.float32))
            # METRICHE
            re = rel_err(vol, reco)
            psnr = PSNR(vol, reco)
            ssim = SSIM(vol.astype(np.float32), reco.astype(np.float32))
            this_exp_metrics.append( (re, psnr, ssim) )
            print(f"Volume {i}: RE={re:.3g} PSNR={psnr:.2f} SSIM={ssim:.3f}")

            # VISUALIZZAZIONE 3D solo su primo volume della serie e GT
            if i==0:
                print('  > Ground Truth (show3d)')
                sparse_vis.show3d(vol)
                print('  > Feldkamp reco (show3d)')
                sparse_vis.show3d(reco)

        this_exp_metrics = np.array(this_exp_metrics) # shape (N_VOLS,3)
        avg_re = np.mean(this_exp_metrics[:,0])
        avg_psnr = np.mean(this_exp_metrics[:,1])
        avg_ssim = np.mean(this_exp_metrics[:,2])
        print(f"[MEDIA {range_name}, noise={nl}]  RE={avg_re:.3g}  PSNR={avg_psnr:.2f}  SSIM={avg_ssim:.3f}")
        results.append( (range_name, nl, avg_re, avg_psnr, avg_ssim) )

# ====== TABELLA FINALE RIASSUNTIVA ======
print("\n======= RISULTATI FINALI MEDIA METRICHE =======")
print("Geometria    | Noise | RE    | PSNR  | SSIM")
for row in results:
    print(f"{row[0]:<12} | {row[1]:.3f} | {row[2]:.3g} | {row[3]:.2f} | {row[4]:.3f}")
