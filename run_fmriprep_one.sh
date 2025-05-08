#!/bin/bash
#SBATCH --job-name=fmriprep_%j
#SBATCH --output=output/fmriprep_%x_%j.out
#SBATCH --error=output/fmriprep_%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G

sub_id=$1

echo "Running fMRIPrep for subject: $sub_id"

mkdir -p output

cd /scratch/users/k21116947/abcd-mproc-release5

singularity run --cleanenv \
  -B /scratch/users/k21116947/fmriprep_work:/work \
  -B /scratch/users/k21116947/abcd-mproc-release5:/data:ro \
  -B /scratch/users/k21116947/derivatives:/out \
  -B /scratch/users/k21116947/license.txt:/opt/freesurfer/license.txt:ro \
  /scratch/users/k21116947/fmriprep-latest.sif \
  /data /out participant \
  --participant-label $sub_id \
  --work-dir /work \
  --fs-license-file /opt/freesurfer/license.txt \
  --ignore fieldmaps \
  --output-spaces MNI152NLin2009cAsym \
  --nthreads 16 --omp-nthreads 8 --mem_mb 64000

