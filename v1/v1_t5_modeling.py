{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "359261be-ae66-4abc-9fc8-2a6c814057ed",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/data/ephemeral/home/py310/lib/python3.10/site-packages/transformers/utils/generic.py:441: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.\n",
      "  _torch_pytree._register_pytree_node(\n",
      "/data/ephemeral/home/py310/lib/python3.10/site-packages/transformers/utils/generic.py:309: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.\n",
      "  _torch_pytree._register_pytree_node(\n",
      "/data/ephemeral/home/py310/lib/python3.10/site-packages/transformers/utils/generic.py:309: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.\n",
      "  _torch_pytree._register_pytree_node(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "===== DEBUG: Dataset Check =====\n",
      "Train size: 11211\n",
      "Valid size: 1246\n",
      "=================================\n",
      "\n",
      "\n",
      "===== DEBUG: Global GPU =====\n",
      "GPU: NVIDIA GeForce RTX 3090\n",
      "========================================\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/data/ephemeral/home/py310/lib/python3.10/site-packages/huggingface_hub/file_download.py:942: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.\n",
      "  warnings.warn(\n",
      "[I 2025-12-02 01:03:43,854] A new study created in memory with name: no-name-bda755e5-4864-4203-8e98-07baf00f2eb0\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "===== DEBUG: Tokenization Preview =====\n",
      "Input token length: 163\n",
      "Target token length: 24\n",
      "=======================================\n",
      "\n",
      "🔥 Optuna Search 시작!\n",
      "\n",
      "===== DEBUG: Trial 0 / lr=3e-05, warm=0.15, ep=4 =====\n",
      "\n",
      "\n",
      "===== DEBUG: Tokenizer / Model =====\n",
      "Special tokens added: 19\n",
      "Tokenizer vocab size: 50377\n",
      "Model embedding size: 50377\n",
      "=====================================\n",
      "\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "337ad8adde9c41fdb4f622c0e9f47d80",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Map:   0%|          | 0/11211 [00:00<?, ? examples/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8d80660ecfcf4d479ce3151269c5d3af",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Map:   0%|          | 0/1246 [00:00<?, ? examples/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[34m\u001b[1mwandb\u001b[0m: Currently logged in as: \u001b[33mmilpasoomin\u001b[0m (\u001b[33mmilpasoomin-no\u001b[0m). Use \u001b[1m`wandb login --relogin`\u001b[0m to force relogin\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "wandb version 0.23.0 is available!  To upgrade, please run:\n",
       " $ pip install wandb --upgrade"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Tracking run with wandb version 0.16.1"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Run data is saved locally in <code>/root/nlp/notebooks/modeling/wandb/run-20251202_010355-7m35agjx</code>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Syncing run <strong><a href='https://wandb.ai/milpasoomin-no/fastcampus_text_generation_with_transformer/runs/7m35agjx' target=\"_blank\">v1_t5_large_trial0</a></strong> to <a href='https://wandb.ai/milpasoomin-no/fastcampus_text_generation_with_transformer' target=\"_blank\">Weights & Biases</a> (<a href='https://wandb.me/run' target=\"_blank\">docs</a>)<br/>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View project at <a href='https://wandb.ai/milpasoomin-no/fastcampus_text_generation_with_transformer' target=\"_blank\">https://wandb.ai/milpasoomin-no/fastcampus_text_generation_with_transformer</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View run at <a href='https://wandb.ai/milpasoomin-no/fastcampus_text_generation_with_transformer/runs/7m35agjx' target=\"_blank\">https://wandb.ai/milpasoomin-no/fastcampus_text_generation_with_transformer/runs/7m35agjx</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/data/ephemeral/home/py310/lib/python3.10/site-packages/accelerate/accelerator.py:451: FutureWarning: Passing the following arguments to `Accelerator` is deprecated and will be removed in version 1.0 of Accelerate: dict_keys(['dispatch_batches', 'split_batches']). Please pass an `accelerate.DataLoaderConfiguration` instead: \n",
      "dataloader_config = DataLoaderConfiguration(dispatch_batches=None, split_batches=False)\n",
      "  warnings.warn(\n",
      "Detected kernel version 5.4.0, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "\n",
       "    <div>\n",
       "      \n",
       "      <progress value='5604' max='5604' style='width:300px; height:20px; vertical-align: middle;'></progress>\n",
       "      [5604/5604 1:30:06, Epoch 3/4]\n",
       "    </div>\n",
       "    <table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       " <tr style=\"text-align: left;\">\n",
       "      <th>Step</th>\n",
       "      <th>Training Loss</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>100</td>\n",
       "      <td>3.900000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>200</td>\n",
       "      <td>1.595600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>300</td>\n",
       "      <td>1.126900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>400</td>\n",
       "      <td>1.038500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>500</td>\n",
       "      <td>0.988000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>600</td>\n",
       "      <td>0.983300</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>700</td>\n",
       "      <td>0.954900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>800</td>\n",
       "      <td>0.931900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>900</td>\n",
       "      <td>0.918100</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1000</td>\n",
       "      <td>0.916800</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1100</td>\n",
       "      <td>0.884600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1200</td>\n",
       "      <td>0.882300</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1300</td>\n",
       "      <td>0.876900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1400</td>\n",
       "      <td>0.871700</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1500</td>\n",
       "      <td>0.812300</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1600</td>\n",
       "      <td>0.804100</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1700</td>\n",
       "      <td>0.798900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1800</td>\n",
       "      <td>0.796900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1900</td>\n",
       "      <td>0.797600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2000</td>\n",
       "      <td>0.790500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2100</td>\n",
       "      <td>0.795300</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2200</td>\n",
       "      <td>0.775300</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2300</td>\n",
       "      <td>0.779100</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2400</td>\n",
       "      <td>0.779900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2500</td>\n",
       "      <td>0.774300</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2600</td>\n",
       "      <td>0.790700</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2700</td>\n",
       "      <td>0.779800</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2800</td>\n",
       "      <td>0.770500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2900</td>\n",
       "      <td>0.699600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3000</td>\n",
       "      <td>0.706400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3100</td>\n",
       "      <td>0.696700</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3200</td>\n",
       "      <td>0.719300</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3300</td>\n",
       "      <td>0.707400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3400</td>\n",
       "      <td>0.691900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3500</td>\n",
       "      <td>0.697000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3600</td>\n",
       "      <td>0.699300</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3700</td>\n",
       "      <td>0.703900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3800</td>\n",
       "      <td>0.703500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3900</td>\n",
       "      <td>0.703900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4000</td>\n",
       "      <td>0.699400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4100</td>\n",
       "      <td>0.686400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4200</td>\n",
       "      <td>0.693600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4300</td>\n",
       "      <td>0.655900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4400</td>\n",
       "      <td>0.661400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4500</td>\n",
       "      <td>0.654600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4600</td>\n",
       "      <td>0.655000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4700</td>\n",
       "      <td>0.644100</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4800</td>\n",
       "      <td>0.636100</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4900</td>\n",
       "      <td>0.655000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>5000</td>\n",
       "      <td>0.649100</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>5100</td>\n",
       "      <td>0.656600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>5200</td>\n",
       "      <td>0.644300</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>5300</td>\n",
       "      <td>0.653100</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>5400</td>\n",
       "      <td>0.665600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>5500</td>\n",
       "      <td>0.633600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>5600</td>\n",
       "      <td>0.646100</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table><p>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/data/ephemeral/home/py310/lib/python3.10/site-packages/transformers/generation/utils.py:1273: UserWarning: Using the model-agnostic default `max_length` (=20) to control the generation length. We recommend setting `max_new_tokens` to control the maximum length of the generation.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/html": [],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "⚡ compute_metrics CALLED!\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "VBox(children=(Label(value='0.006 MB of 0.006 MB uploaded\\r'), FloatProgress(value=1.0, max=1.0)))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<style>\n",
       "    table.wandb td:nth-child(1) { padding: 0 10px; text-align: left ; width: auto;} td:nth-child(2) {text-align: left ; width: 100%}\n",
       "    .wandb-row { display: flex; flex-direction: row; flex-wrap: wrap; justify-content: flex-start; width: 100% }\n",
       "    .wandb-col { display: flex; flex-direction: column; flex-basis: 100%; flex: 1; padding: 10px; }\n",
       "    </style>\n",
       "<div class=\"wandb-row\"><div class=\"wandb-col\"><h3>Run history:</h3><br/><table class=\"wandb\"><tr><td>final_rouge</td><td>▁</td></tr><tr><td>train/epoch</td><td>▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███</td></tr><tr><td>train/global_step</td><td>▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇████</td></tr><tr><td>train/learning_rate</td><td>▂▃▄▅▆████▇▇▇▇▇▆▆▆▆▅▅▅▅▅▄▄▄▄▄▃▃▃▃▂▂▂▂▂▁▁▁</td></tr><tr><td>train/loss</td><td>█▃▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁</td></tr><tr><td>train/total_flos</td><td>▁</td></tr><tr><td>train/train_loss</td><td>▁</td></tr><tr><td>train/train_runtime</td><td>▁</td></tr><tr><td>train/train_samples_per_second</td><td>▁</td></tr><tr><td>train/train_steps_per_second</td><td>▁</td></tr></table><br/></div><div class=\"wandb-col\"><h3>Run summary:</h3><br/><table class=\"wandb\"><tr><td>final_rouge</td><td>1.50294</td></tr><tr><td>train/epoch</td><td>4.0</td></tr><tr><td>train/global_step</td><td>5604</td></tr><tr><td>train/learning_rate</td><td>0.0</td></tr><tr><td>train/loss</td><td>0.6461</td></tr><tr><td>train/total_flos</td><td>1.0589391353806848e+17</td></tr><tr><td>train/train_loss</td><td>0.83616</td></tr><tr><td>train/train_runtime</td><td>5408.1409</td></tr><tr><td>train/train_samples_per_second</td><td>8.292</td></tr><tr><td>train/train_steps_per_second</td><td>1.036</td></tr></table><br/></div></div>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View run <strong style=\"color:#cdcd00\">v1_t5_large_trial0</strong> at: <a href='https://wandb.ai/milpasoomin-no/fastcampus_text_generation_with_transformer/runs/7m35agjx' target=\"_blank\">https://wandb.ai/milpasoomin-no/fastcampus_text_generation_with_transformer/runs/7m35agjx</a><br/>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Find logs at: <code>./wandb/run-20251202_010355-7m35agjx/logs</code>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-12-02 02:36:26,268] Trial 0 finished with value: 1.5029391043235345 and parameters: {'learning_rate': '3e-5', 'warmup_ratio': 0.15, 'num_train_epochs': 4}. Best is trial 0 with value: 1.5029391043235345.\n",
      "/data/ephemeral/home/py310/lib/python3.10/site-packages/huggingface_hub/file_download.py:942: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "===== DEBUG: Trial 1 / lr=3e-05, warm=0.1, ep=3 =====\n",
      "\n",
      "\n",
      "===== DEBUG: Tokenizer / Model =====\n",
      "Special tokens added: 19\n",
      "Tokenizer vocab size: 50377\n",
      "Model embedding size: 50377\n",
      "=====================================\n",
      "\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a196dee2e0c94d2bac8bf2b601fea241",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Map:   0%|          | 0/11211 [00:00<?, ? examples/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "82e13031c8694d9882646a4a2c1568a6",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Map:   0%|          | 0/1246 [00:00<?, ? examples/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "wandb version 0.23.0 is available!  To upgrade, please run:\n",
       " $ pip install wandb --upgrade"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Tracking run with wandb version 0.16.1"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Run data is saved locally in <code>/root/nlp/notebooks/modeling/wandb/run-20251202_023635-vcyg408a</code>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Syncing run <strong><a href='https://wandb.ai/milpasoomin-no/fastcampus_text_generation_with_transformer/runs/vcyg408a' target=\"_blank\">v1_t5_large_trial1</a></strong> to <a href='https://wandb.ai/milpasoomin-no/fastcampus_text_generation_with_transformer' target=\"_blank\">Weights & Biases</a> (<a href='https://wandb.me/run' target=\"_blank\">docs</a>)<br/>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View project at <a href='https://wandb.ai/milpasoomin-no/fastcampus_text_generation_with_transformer' target=\"_blank\">https://wandb.ai/milpasoomin-no/fastcampus_text_generation_with_transformer</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View run at <a href='https://wandb.ai/milpasoomin-no/fastcampus_text_generation_with_transformer/runs/vcyg408a' target=\"_blank\">https://wandb.ai/milpasoomin-no/fastcampus_text_generation_with_transformer/runs/vcyg408a</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/data/ephemeral/home/py310/lib/python3.10/site-packages/accelerate/accelerator.py:451: FutureWarning: Passing the following arguments to `Accelerator` is deprecated and will be removed in version 1.0 of Accelerate: dict_keys(['dispatch_batches', 'split_batches']). Please pass an `accelerate.DataLoaderConfiguration` instead: \n",
      "dataloader_config = DataLoaderConfiguration(dispatch_batches=None, split_batches=False)\n",
      "  warnings.warn(\n",
      "WARNING:accelerate.utils.other:Detected kernel version 5.4.0, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "\n",
       "    <div>\n",
       "      \n",
       "      <progress value='4203' max='4203' style='width:300px; height:20px; vertical-align: middle;'></progress>\n",
       "      [4203/4203 1:07:55, Epoch 2/3]\n",
       "    </div>\n",
       "    <table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       " <tr style=\"text-align: left;\">\n",
       "      <th>Step</th>\n",
       "      <th>Training Loss</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>100</td>\n",
       "      <td>3.337200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>200</td>\n",
       "      <td>1.211900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>300</td>\n",
       "      <td>1.051200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>400</td>\n",
       "      <td>0.986400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>500</td>\n",
       "      <td>0.944700</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>600</td>\n",
       "      <td>0.945600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>700</td>\n",
       "      <td>0.922900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>800</td>\n",
       "      <td>0.903500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>900</td>\n",
       "      <td>0.894300</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1000</td>\n",
       "      <td>0.897300</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1100</td>\n",
       "      <td>0.866900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1200</td>\n",
       "      <td>0.866600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1300</td>\n",
       "      <td>0.862000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1400</td>\n",
       "      <td>0.857400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1500</td>\n",
       "      <td>0.792700</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1600</td>\n",
       "      <td>0.785700</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1700</td>\n",
       "      <td>0.782100</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1800</td>\n",
       "      <td>0.781900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1900</td>\n",
       "      <td>0.786000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2000</td>\n",
       "      <td>0.780100</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2100</td>\n",
       "      <td>0.785600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2200</td>\n",
       "      <td>0.765400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2300</td>\n",
       "      <td>0.770600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2400</td>\n",
       "      <td>0.770400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2500</td>\n",
       "      <td>0.765900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2600</td>\n",
       "      <td>0.781800</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2700</td>\n",
       "      <td>0.771000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2800</td>\n",
       "      <td>0.761000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2900</td>\n",
       "      <td>0.706400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3000</td>\n",
       "      <td>0.712500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3100</td>\n",
       "      <td>0.703800</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3200</td>\n",
       "      <td>0.726300</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3300</td>\n",
       "      <td>0.714800</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3400</td>\n",
       "      <td>0.699500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3500</td>\n",
       "      <td>0.701000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3600</td>\n",
       "      <td>0.706800</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3700</td>\n",
       "      <td>0.712400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3800</td>\n",
       "      <td>0.708500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3900</td>\n",
       "      <td>0.711300</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4000</td>\n",
       "      <td>0.708200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4100</td>\n",
       "      <td>0.694400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4200</td>\n",
       "      <td>0.701600</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table><p>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/data/ephemeral/home/py310/lib/python3.10/site-packages/transformers/generation/utils.py:1273: UserWarning: Using the model-agnostic default `max_length` (=20) to control the generation length. We recommend setting `max_new_tokens` to control the maximum length of the generation.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/html": [],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "⚡ compute_metrics CALLED!\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "VBox(children=(Label(value='0.006 MB of 0.006 MB uploaded\\r'), FloatProgress(value=1.0, max=1.0)))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<style>\n",
       "    table.wandb td:nth-child(1) { padding: 0 10px; text-align: left ; width: auto;} td:nth-child(2) {text-align: left ; width: 100%}\n",
       "    .wandb-row { display: flex; flex-direction: row; flex-wrap: wrap; justify-content: flex-start; width: 100% }\n",
       "    .wandb-col { display: flex; flex-direction: column; flex-basis: 100%; flex: 1; padding: 10px; }\n",
       "    </style>\n",
       "<div class=\"wandb-row\"><div class=\"wandb-col\"><h3>Run history:</h3><br/><table class=\"wandb\"><tr><td>final_rouge</td><td>▁</td></tr><tr><td>train/epoch</td><td>▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇███</td></tr><tr><td>train/global_step</td><td>▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▇▇▇▇▇▇████</td></tr><tr><td>train/learning_rate</td><td>▃▄▆████▇▇▇▇▇▆▆▆▆▆▆▅▅▅▅▄▄▄▄▄▃▃▃▃▃▃▂▂▂▂▂▁▁</td></tr><tr><td>train/loss</td><td>█▂▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁</td></tr><tr><td>train/total_flos</td><td>▁</td></tr><tr><td>train/train_loss</td><td>▁</td></tr><tr><td>train/train_runtime</td><td>▁</td></tr><tr><td>train/train_samples_per_second</td><td>▁</td></tr><tr><td>train/train_steps_per_second</td><td>▁</td></tr></table><br/></div><div class=\"wandb-col\"><h3>Run summary:</h3><br/><table class=\"wandb\"><tr><td>final_rouge</td><td>1.473</td></tr><tr><td>train/epoch</td><td>3.0</td></tr><tr><td>train/global_step</td><td>4203</td></tr><tr><td>train/learning_rate</td><td>0.0</td></tr><tr><td>train/loss</td><td>0.7016</td></tr><tr><td>train/total_flos</td><td>7.942102569713664e+16</td></tr><tr><td>train/train_loss</td><td>0.86497</td></tr><tr><td>train/train_runtime</td><td>4076.2917</td></tr><tr><td>train/train_samples_per_second</td><td>8.251</td></tr><tr><td>train/train_steps_per_second</td><td>1.031</td></tr></table><br/></div></div>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View run <strong style=\"color:#cdcd00\">v1_t5_large_trial1</strong> at: <a href='https://wandb.ai/milpasoomin-no/fastcampus_text_generation_with_transformer/runs/vcyg408a' target=\"_blank\">https://wandb.ai/milpasoomin-no/fastcampus_text_generation_with_transformer/runs/vcyg408a</a><br/>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Find logs at: <code>./wandb/run-20251202_023635-vcyg408a/logs</code>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[I 2025-12-02 03:46:54,064] Trial 1 finished with value: 1.4730004590558363 and parameters: {'learning_rate': '3e-5', 'warmup_ratio': 0.1, 'num_train_epochs': 3}. Best is trial 0 with value: 1.5029391043235345.\n",
      "/data/ephemeral/home/py310/lib/python3.10/site-packages/huggingface_hub/file_download.py:942: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "🎉 Best Trial: {'learning_rate': '3e-5', 'warmup_ratio': 0.15, 'num_train_epochs': 4}\n",
      "\n",
      "🔧 Seed Ensemble 시작 — seeds = [42, 2025]\n",
      "\n",
      "===== 🚀 Seed 42 =====\n",
      "\n",
      "===== DEBUG: Tokenizer / Model =====\n",
      "Special tokens added: 19\n",
      "Tokenizer vocab size: 50377\n",
      "Model embedding size: 50377\n",
      "=====================================\n",
      "\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "78e455de166749b1bd7c54844b1dd636",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Map:   0%|          | 0/11211 [00:00<?, ? examples/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1850fda8a9be4aeca0dd88409b025ad3",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Map:   0%|          | 0/1246 [00:00<?, ? examples/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "wandb version 0.23.0 is available!  To upgrade, please run:\n",
       " $ pip install wandb --upgrade"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Tracking run with wandb version 0.16.1"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Run data is saved locally in <code>/root/nlp/notebooks/modeling/wandb/run-20251202_034703-5xe3w8xl</code>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Syncing run <strong><a href='https://wandb.ai/milpasoomin-no/fastcampus_text_generation_with_transformer/runs/5xe3w8xl' target=\"_blank\">v1_t5_large_seed42</a></strong> to <a href='https://wandb.ai/milpasoomin-no/fastcampus_text_generation_with_transformer' target=\"_blank\">Weights & Biases</a> (<a href='https://wandb.me/run' target=\"_blank\">docs</a>)<br/>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View project at <a href='https://wandb.ai/milpasoomin-no/fastcampus_text_generation_with_transformer' target=\"_blank\">https://wandb.ai/milpasoomin-no/fastcampus_text_generation_with_transformer</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View run at <a href='https://wandb.ai/milpasoomin-no/fastcampus_text_generation_with_transformer/runs/5xe3w8xl' target=\"_blank\">https://wandb.ai/milpasoomin-no/fastcampus_text_generation_with_transformer/runs/5xe3w8xl</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/data/ephemeral/home/py310/lib/python3.10/site-packages/accelerate/accelerator.py:451: FutureWarning: Passing the following arguments to `Accelerator` is deprecated and will be removed in version 1.0 of Accelerate: dict_keys(['dispatch_batches', 'split_batches']). Please pass an `accelerate.DataLoaderConfiguration` instead: \n",
      "dataloader_config = DataLoaderConfiguration(dispatch_batches=None, split_batches=False)\n",
      "  warnings.warn(\n",
      "WARNING:accelerate.utils.other:Detected kernel version 5.4.0, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.\n",
      "You're using a T5TokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "\n",
       "    <div>\n",
       "      \n",
       "      <progress value='5604' max='5604' style='width:300px; height:20px; vertical-align: middle;'></progress>\n",
       "      [5604/5604 1:30:37, Epoch 3/4]\n",
       "    </div>\n",
       "    <table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       " <tr style=\"text-align: left;\">\n",
       "      <th>Step</th>\n",
       "      <th>Training Loss</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>100</td>\n",
       "      <td>3.875900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>200</td>\n",
       "      <td>1.606100</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>300</td>\n",
       "      <td>1.124500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>400</td>\n",
       "      <td>1.036900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>500</td>\n",
       "      <td>0.987500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>600</td>\n",
       "      <td>0.982300</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>700</td>\n",
       "      <td>0.954100</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>800</td>\n",
       "      <td>0.931600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>900</td>\n",
       "      <td>0.917900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1000</td>\n",
       "      <td>0.915900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1100</td>\n",
       "      <td>0.884200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1200</td>\n",
       "      <td>0.881900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1300</td>\n",
       "      <td>0.877000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1400</td>\n",
       "      <td>0.871600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1500</td>\n",
       "      <td>0.812900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1600</td>\n",
       "      <td>0.805000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1700</td>\n",
       "      <td>0.801500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1800</td>\n",
       "      <td>0.798900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1900</td>\n",
       "      <td>0.799000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2000</td>\n",
       "      <td>0.792200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2100</td>\n",
       "      <td>0.796100</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2200</td>\n",
       "      <td>0.776600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2300</td>\n",
       "      <td>0.781100</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2400</td>\n",
       "      <td>0.780200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2500</td>\n",
       "      <td>0.775900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2600</td>\n",
       "      <td>0.791200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2700</td>\n",
       "      <td>0.780300</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2800</td>\n",
       "      <td>0.771000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2900</td>\n",
       "      <td>0.700700</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3000</td>\n",
       "      <td>0.706000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3100</td>\n",
       "      <td>0.697700</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3200</td>\n",
       "      <td>0.719300</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3300</td>\n",
       "      <td>0.707500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3400</td>\n",
       "      <td>0.692800</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3500</td>\n",
       "      <td>0.696600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3600</td>\n",
       "      <td>0.700500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3700</td>\n",
       "      <td>0.705000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3800</td>\n",
       "      <td>0.703300</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3900</td>\n",
       "      <td>0.704400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4000</td>\n",
       "      <td>0.699900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4100</td>\n",
       "      <td>0.686900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4200</td>\n",
       "      <td>0.693500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4300</td>\n",
       "      <td>0.655200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4400</td>\n",
       "      <td>0.660900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4500</td>\n",
       "      <td>0.654400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4600</td>\n",
       "      <td>0.655500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4700</td>\n",
       "      <td>0.643400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4800</td>\n",
       "      <td>0.636000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4900</td>\n",
       "      <td>0.654400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>5000</td>\n",
       "      <td>0.649000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>5100</td>\n",
       "      <td>0.656700</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>5200</td>\n",
       "      <td>0.645000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>5300</td>\n",
       "      <td>0.652200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>5400</td>\n",
       "      <td>0.665200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>5500</td>\n",
       "      <td>0.634300</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>5600</td>\n",
       "      <td>0.646100</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table><p>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/data/ephemeral/home/py310/lib/python3.10/site-packages/transformers/generation/utils.py:1273: UserWarning: Using the model-agnostic default `max_length` (=20) to control the generation length. We recommend setting `max_new_tokens` to control the maximum length of the generation.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "\n",
       "    <div>\n",
       "      \n",
       "      <progress value='156' max='156' style='width:300px; height:20px; vertical-align: middle;'></progress>\n",
       "      [156/156 02:12]\n",
       "    </div>\n",
       "    "
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "⚡ compute_metrics CALLED!\n",
      "📊 Seed 42 Eval: {'eval_loss': 0.7677160501480103, 'eval_final_rouge': 1.4850526613568347, 'eval_runtime': 133.7743, 'eval_samples_per_second': 9.314, 'eval_steps_per_second': 1.166, 'epoch': 4.0}\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "VBox(children=(Label(value='0.006 MB of 0.006 MB uploaded\\r'), FloatProgress(value=1.0, max=1.0)))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<style>\n",
       "    table.wandb td:nth-child(1) { padding: 0 10px; text-align: left ; width: auto;} td:nth-child(2) {text-align: left ; width: 100%}\n",
       "    .wandb-row { display: flex; flex-direction: row; flex-wrap: wrap; justify-content: flex-start; width: 100% }\n",
       "    .wandb-col { display: flex; flex-direction: column; flex-basis: 100%; flex: 1; padding: 10px; }\n",
       "    </style>\n",
       "<div class=\"wandb-row\"><div class=\"wandb-col\"><h3>Run history:</h3><br/><table class=\"wandb\"><tr><td>eval/final_rouge</td><td>▁</td></tr><tr><td>eval/loss</td><td>▁</td></tr><tr><td>eval/runtime</td><td>▁</td></tr><tr><td>eval/samples_per_second</td><td>▁</td></tr><tr><td>eval/steps_per_second</td><td>▁</td></tr><tr><td>train/epoch</td><td>▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇████</td></tr><tr><td>train/global_step</td><td>▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇████</td></tr><tr><td>train/learning_rate</td><td>▂▃▄▅▆████▇▇▇▇▇▆▆▆▆▅▅▅▅▅▄▄▄▄▄▃▃▃▃▂▂▂▂▂▁▁▁</td></tr><tr><td>train/loss</td><td>█▃▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁</td></tr><tr><td>train/total_flos</td><td>▁</td></tr><tr><td>train/train_loss</td><td>▁</td></tr><tr><td>train/train_runtime</td><td>▁</td></tr><tr><td>train/train_samples_per_second</td><td>▁</td></tr><tr><td>train/train_steps_per_second</td><td>▁</td></tr></table><br/></div><div class=\"wandb-col\"><h3>Run summary:</h3><br/><table class=\"wandb\"><tr><td>eval/final_rouge</td><td>1.48505</td></tr><tr><td>eval/loss</td><td>0.76772</td></tr><tr><td>eval/runtime</td><td>133.7743</td></tr><tr><td>eval/samples_per_second</td><td>9.314</td></tr><tr><td>eval/steps_per_second</td><td>1.166</td></tr><tr><td>train/epoch</td><td>4.0</td></tr><tr><td>train/global_step</td><td>5604</td></tr><tr><td>train/learning_rate</td><td>0.0</td></tr><tr><td>train/loss</td><td>0.6461</td></tr><tr><td>train/total_flos</td><td>1.0589391353806848e+17</td></tr><tr><td>train/train_loss</td><td>0.83613</td></tr><tr><td>train/train_runtime</td><td>5438.8</td></tr><tr><td>train/train_samples_per_second</td><td>8.245</td></tr><tr><td>train/train_steps_per_second</td><td>1.03</td></tr></table><br/></div></div>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View run <strong style=\"color:#cdcd00\">v1_t5_large_seed42</strong> at: <a href='https://wandb.ai/milpasoomin-no/fastcampus_text_generation_with_transformer/runs/5xe3w8xl' target=\"_blank\">https://wandb.ai/milpasoomin-no/fastcampus_text_generation_with_transformer/runs/5xe3w8xl</a><br/>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Find logs at: <code>./wandb/run-20251202_034703-5xe3w8xl/logs</code>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "===== 🚀 Seed 2025 =====\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/data/ephemeral/home/py310/lib/python3.10/site-packages/huggingface_hub/file_download.py:942: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "===== DEBUG: Tokenizer / Model =====\n",
      "Special tokens added: 19\n",
      "Tokenizer vocab size: 50377\n",
      "Model embedding size: 50377\n",
      "=====================================\n",
      "\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1917548d85e04befb802f8c3ee109f4a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Map:   0%|          | 0/11211 [00:00<?, ? examples/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "8f7fc05fc82542e1b7f70a5a85d77249",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Map:   0%|          | 0/1246 [00:00<?, ? examples/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "wandb version 0.23.0 is available!  To upgrade, please run:\n",
       " $ pip install wandb --upgrade"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Tracking run with wandb version 0.16.1"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Run data is saved locally in <code>/root/nlp/notebooks/modeling/wandb/run-20251202_052012-ovl9vs9r</code>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Syncing run <strong><a href='https://wandb.ai/milpasoomin-no/fastcampus_text_generation_with_transformer/runs/ovl9vs9r' target=\"_blank\">v1_t5_large_seed2025</a></strong> to <a href='https://wandb.ai/milpasoomin-no/fastcampus_text_generation_with_transformer' target=\"_blank\">Weights & Biases</a> (<a href='https://wandb.me/run' target=\"_blank\">docs</a>)<br/>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View project at <a href='https://wandb.ai/milpasoomin-no/fastcampus_text_generation_with_transformer' target=\"_blank\">https://wandb.ai/milpasoomin-no/fastcampus_text_generation_with_transformer</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View run at <a href='https://wandb.ai/milpasoomin-no/fastcampus_text_generation_with_transformer/runs/ovl9vs9r' target=\"_blank\">https://wandb.ai/milpasoomin-no/fastcampus_text_generation_with_transformer/runs/ovl9vs9r</a>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/data/ephemeral/home/py310/lib/python3.10/site-packages/accelerate/accelerator.py:451: FutureWarning: Passing the following arguments to `Accelerator` is deprecated and will be removed in version 1.0 of Accelerate: dict_keys(['dispatch_batches', 'split_batches']). Please pass an `accelerate.DataLoaderConfiguration` instead: \n",
      "dataloader_config = DataLoaderConfiguration(dispatch_batches=None, split_batches=False)\n",
      "  warnings.warn(\n",
      "WARNING:accelerate.utils.other:Detected kernel version 5.4.0, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.\n",
      "You're using a T5TokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "\n",
       "    <div>\n",
       "      \n",
       "      <progress value='5604' max='5604' style='width:300px; height:20px; vertical-align: middle;'></progress>\n",
       "      [5604/5604 1:30:38, Epoch 3/4]\n",
       "    </div>\n",
       "    <table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       " <tr style=\"text-align: left;\">\n",
       "      <th>Step</th>\n",
       "      <th>Training Loss</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>100</td>\n",
       "      <td>3.879600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>200</td>\n",
       "      <td>1.577900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>300</td>\n",
       "      <td>1.123100</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>400</td>\n",
       "      <td>1.034500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>500</td>\n",
       "      <td>0.986000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>600</td>\n",
       "      <td>0.981400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>700</td>\n",
       "      <td>0.954300</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>800</td>\n",
       "      <td>0.931600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>900</td>\n",
       "      <td>0.917500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1000</td>\n",
       "      <td>0.915900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1100</td>\n",
       "      <td>0.883500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1200</td>\n",
       "      <td>0.881800</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1300</td>\n",
       "      <td>0.876400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1400</td>\n",
       "      <td>0.870700</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1500</td>\n",
       "      <td>0.810600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1600</td>\n",
       "      <td>0.802200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1700</td>\n",
       "      <td>0.797300</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1800</td>\n",
       "      <td>0.795900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1900</td>\n",
       "      <td>0.796500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2000</td>\n",
       "      <td>0.788800</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2100</td>\n",
       "      <td>0.794200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2200</td>\n",
       "      <td>0.774300</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2300</td>\n",
       "      <td>0.779000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2400</td>\n",
       "      <td>0.779000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2500</td>\n",
       "      <td>0.774600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2600</td>\n",
       "      <td>0.789700</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2700</td>\n",
       "      <td>0.779900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2800</td>\n",
       "      <td>0.768900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2900</td>\n",
       "      <td>0.699300</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3000</td>\n",
       "      <td>0.705000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3100</td>\n",
       "      <td>0.697200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3200</td>\n",
       "      <td>0.718200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3300</td>\n",
       "      <td>0.706900</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3400</td>\n",
       "      <td>0.692000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3500</td>\n",
       "      <td>0.695800</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3600</td>\n",
       "      <td>0.699200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3700</td>\n",
       "      <td>0.703800</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3800</td>\n",
       "      <td>0.702800</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3900</td>\n",
       "      <td>0.702800</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4000</td>\n",
       "      <td>0.699000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4100</td>\n",
       "      <td>0.686200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4200</td>\n",
       "      <td>0.692000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4300</td>\n",
       "      <td>0.654600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4400</td>\n",
       "      <td>0.660500</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4500</td>\n",
       "      <td>0.653600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4600</td>\n",
       "      <td>0.654200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4700</td>\n",
       "      <td>0.642700</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4800</td>\n",
       "      <td>0.636300</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4900</td>\n",
       "      <td>0.653200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>5000</td>\n",
       "      <td>0.647800</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>5100</td>\n",
       "      <td>0.655300</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>5200</td>\n",
       "      <td>0.643700</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>5300</td>\n",
       "      <td>0.651600</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>5400</td>\n",
       "      <td>0.664100</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>5500</td>\n",
       "      <td>0.633400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>5600</td>\n",
       "      <td>0.645800</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table><p>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/data/ephemeral/home/py310/lib/python3.10/site-packages/transformers/generation/utils.py:1273: UserWarning: Using the model-agnostic default `max_length` (=20) to control the generation length. We recommend setting `max_new_tokens` to control the maximum length of the generation.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "\n",
       "    <div>\n",
       "      \n",
       "      <progress value='156' max='156' style='width:300px; height:20px; vertical-align: middle;'></progress>\n",
       "      [156/156 02:12]\n",
       "    </div>\n",
       "    "
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "⚡ compute_metrics CALLED!\n",
      "📊 Seed 2025 Eval: {'eval_loss': 0.7675471901893616, 'eval_final_rouge': 1.4809016001833017, 'eval_runtime': 133.3177, 'eval_samples_per_second': 9.346, 'eval_steps_per_second': 1.17, 'epoch': 4.0}\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "VBox(children=(Label(value='0.006 MB of 0.006 MB uploaded\\r'), FloatProgress(value=1.0, max=1.0)))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<style>\n",
       "    table.wandb td:nth-child(1) { padding: 0 10px; text-align: left ; width: auto;} td:nth-child(2) {text-align: left ; width: 100%}\n",
       "    .wandb-row { display: flex; flex-direction: row; flex-wrap: wrap; justify-content: flex-start; width: 100% }\n",
       "    .wandb-col { display: flex; flex-direction: column; flex-basis: 100%; flex: 1; padding: 10px; }\n",
       "    </style>\n",
       "<div class=\"wandb-row\"><div class=\"wandb-col\"><h3>Run history:</h3><br/><table class=\"wandb\"><tr><td>eval/final_rouge</td><td>▁</td></tr><tr><td>eval/loss</td><td>▁</td></tr><tr><td>eval/runtime</td><td>▁</td></tr><tr><td>eval/samples_per_second</td><td>▁</td></tr><tr><td>eval/steps_per_second</td><td>▁</td></tr><tr><td>train/epoch</td><td>▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇████</td></tr><tr><td>train/global_step</td><td>▁▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▆▆▆▆▆▆▇▇▇▇▇████</td></tr><tr><td>train/learning_rate</td><td>▂▃▄▅▆████▇▇▇▇▇▆▆▆▆▅▅▅▅▅▄▄▄▄▄▃▃▃▃▂▂▂▂▂▁▁▁</td></tr><tr><td>train/loss</td><td>█▃▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁</td></tr><tr><td>train/total_flos</td><td>▁</td></tr><tr><td>train/train_loss</td><td>▁</td></tr><tr><td>train/train_runtime</td><td>▁</td></tr><tr><td>train/train_samples_per_second</td><td>▁</td></tr><tr><td>train/train_steps_per_second</td><td>▁</td></tr></table><br/></div><div class=\"wandb-col\"><h3>Run summary:</h3><br/><table class=\"wandb\"><tr><td>eval/final_rouge</td><td>1.4809</td></tr><tr><td>eval/loss</td><td>0.76755</td></tr><tr><td>eval/runtime</td><td>133.3177</td></tr><tr><td>eval/samples_per_second</td><td>9.346</td></tr><tr><td>eval/steps_per_second</td><td>1.17</td></tr><tr><td>train/epoch</td><td>4.0</td></tr><tr><td>train/global_step</td><td>5604</td></tr><tr><td>train/learning_rate</td><td>0.0</td></tr><tr><td>train/loss</td><td>0.6458</td></tr><tr><td>train/total_flos</td><td>1.0589391353806848e+17</td></tr><tr><td>train/train_loss</td><td>0.83453</td></tr><tr><td>train/train_runtime</td><td>5439.6775</td></tr><tr><td>train/train_samples_per_second</td><td>8.244</td></tr><tr><td>train/train_steps_per_second</td><td>1.03</td></tr></table><br/></div></div>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       " View run <strong style=\"color:#cdcd00\">v1_t5_large_seed2025</strong> at: <a href='https://wandb.ai/milpasoomin-no/fastcampus_text_generation_with_transformer/runs/ovl9vs9r' target=\"_blank\">https://wandb.ai/milpasoomin-no/fastcampus_text_generation_with_transformer/runs/ovl9vs9r</a><br/>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "Find logs at: <code>./wandb/run-20251202_052012-ovl9vs9r/logs</code>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "🎉 전체 Seed Ensemble 완료!\n"
     ]
    }
   ],
   "source": [
    "# ===============================================================\n",
    "# train.py — Optuna + WandB + ROUGE 평가 + Seed Ensemble (완전 수정본)\n",
    "# ===============================================================\n",
    "\n",
    "import os\n",
    "import yaml\n",
    "import wandb\n",
    "import optuna\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import time\n",
    "import torch\n",
    "from datasets import Dataset\n",
    "from functools import partial\n",
    "from rouge_score import rouge_scorer\n",
    "from transformers import set_seed\n",
    "\n",
    "from transformers import (\n",
    "    T5ForConditionalGeneration,\n",
    "    T5TokenizerFast,\n",
    "    Seq2SeqTrainer,\n",
    "    Seq2SeqTrainingArguments,\n",
    ")\n",
    "\n",
    "torch.backends.cuda.matmul.allow_tf32 = True\n",
    "\n",
    "\n",
    "# ===============================================================\n",
    "# 0) Load config.yaml\n",
    "# ===============================================================\n",
    "def load_config(path=\"config.yaml\"):\n",
    "    with open(path, \"r\") as f:\n",
    "        return yaml.safe_load(f)\n",
    "\n",
    "config = load_config()\n",
    "\n",
    "\n",
    "# ===============================================================\n",
    "# 1) Load dataset\n",
    "# ===============================================================\n",
    "def load_train_valid(cfg):\n",
    "    df_path = os.path.join(cfg[\"general\"][\"data_dir\"], cfg[\"general\"][\"train_file\"])\n",
    "    df = pd.read_csv(df_path)\n",
    "\n",
    "    df = df[[\"dialogue_clean\", \"summary\"]].rename(\n",
    "        columns={\"dialogue_clean\": \"input_text\", \"summary\": \"target_text\"}\n",
    "    )\n",
    "\n",
    "    train_df = df.sample(frac=0.9, random_state=cfg[\"general\"][\"seed\"])\n",
    "    valid_df = df.drop(train_df.index)\n",
    "\n",
    "    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))\n",
    "    valid_ds = Dataset.from_pandas(valid_df.reset_index(drop=True))\n",
    "\n",
    "    return train_ds, valid_ds, valid_df.reset_index(drop=True)\n",
    "\n",
    "\n",
    "train_dataset, valid_dataset, valid_df_pandas = load_train_valid(config)\n",
    "\n",
    "\n",
    "print(\"\\n===== DEBUG: Dataset Check =====\")\n",
    "print(\"Train size:\", len(train_dataset))\n",
    "print(\"Valid size:\", len(valid_dataset))\n",
    "print(\"=================================\\n\")\n",
    "\n",
    "\n",
    "# ===============================================================\n",
    "# 2) Tokenizer + Model\n",
    "# ===============================================================\n",
    "def create_tokenizer_and_model(cfg):\n",
    "    tokenizer = T5TokenizerFast.from_pretrained(cfg[\"general\"][\"model_name\"])\n",
    "    num_added = tokenizer.add_tokens(cfg[\"tokenizer\"][\"special_tokens\"])\n",
    "\n",
    "    model = T5ForConditionalGeneration.from_pretrained(cfg[\"general\"][\"model_name\"])\n",
    "    model.resize_token_embeddings(len(tokenizer))\n",
    "\n",
    "    print(\"\\n===== DEBUG: Tokenizer / Model =====\")\n",
    "    print(\"Special tokens added:\", num_added)\n",
    "    print(\"Tokenizer vocab size:\", len(tokenizer))\n",
    "    print(\"Model embedding size:\", model.get_input_embeddings().weight.shape[0])\n",
    "    print(\"=====================================\\n\")\n",
    "\n",
    "    return tokenizer, model\n",
    "\n",
    "\n",
    "# ===============================================================\n",
    "# 3) Preprocess\n",
    "# ===============================================================\n",
    "def preprocess(batch, tokenizer, cfg_tokenizer, prefix=\"\"):\n",
    "    inputs = [prefix + x for x in batch[\"input_text\"]]\n",
    "\n",
    "    enc = tokenizer(\n",
    "        inputs,\n",
    "        max_length=cfg_tokenizer[\"encoder_max_len\"],\n",
    "        truncation=True,\n",
    "        padding=\"max_length\",\n",
    "    )\n",
    "\n",
    "    dec = tokenizer(\n",
    "        batch[\"target_text\"],\n",
    "        max_length=cfg_tokenizer[\"decoder_max_len\"],\n",
    "        truncation=True,\n",
    "        padding=\"max_length\",\n",
    "    )[\"input_ids\"]\n",
    "\n",
    "    pad = tokenizer.pad_token_id\n",
    "    enc[\"labels\"] = [[t if t != pad else -100 for t in seq] for seq in dec]\n",
    "\n",
    "    return enc\n",
    "\n",
    "\n",
    "# ===============================================================\n",
    "# 4) Tokenization Preview\n",
    "# ===============================================================\n",
    "def debug_tokenization_preview(tokenizer, cfg_tokenizer, prefix, dataset):\n",
    "\n",
    "    sample = dataset[0]\n",
    "    input_text = prefix + sample[\"input_text\"]\n",
    "    target_text = sample[\"target_text\"]\n",
    "\n",
    "    enc = tokenizer(input_text, max_length=cfg_tokenizer[\"encoder_max_len\"], truncation=True)\n",
    "    dec = tokenizer(target_text, max_length=cfg_tokenizer[\"decoder_max_len\"], truncation=True)\n",
    "\n",
    "    print(\"\\n===== DEBUG: Tokenization Preview =====\")\n",
    "    print(\"Input token length:\", len(enc[\"input_ids\"]))\n",
    "    print(\"Target token length:\", len(dec[\"input_ids\"]))\n",
    "    print(\"=======================================\\n\")\n",
    "\n",
    "\n",
    "# ===============================================================\n",
    "# 5) ROUGE + compute_metrics\n",
    "# ===============================================================\n",
    "def compute_rouge_scores(pred, refs):\n",
    "    scorer = rouge_scorer.RougeScorer([\"rouge1\", \"rouge2\", \"rougeL\"], use_stemmer=True)\n",
    "\n",
    "    r1 = np.mean([scorer.score(r, pred)[\"rouge1\"].fmeasure for r in refs])\n",
    "    r2 = np.mean([scorer.score(r, pred)[\"rouge2\"].fmeasure for r in refs])\n",
    "    rl = np.mean([scorer.score(r, pred)[\"rougeL\"].fmeasure for r in refs])\n",
    "\n",
    "    return float(r1 + r2 + rl)\n",
    "\n",
    "\n",
    "def compute_metrics(eval_pred, tokenizer, gold_df):\n",
    "    print(\"⚡ compute_metrics CALLED!\")\n",
    "    preds, labels = eval_pred\n",
    "\n",
    "    if preds.ndim == 3:\n",
    "        preds = preds.argmax(-1)\n",
    "\n",
    "    preds = np.clip(preds.astype(np.int64), 0, tokenizer.vocab_size - 1)\n",
    "\n",
    "    decoded = tokenizer.batch_decode(preds, skip_special_tokens=True)\n",
    "    decoded = [d.strip() for d in decoded]\n",
    "\n",
    "    scores = []\n",
    "    for i, pred in enumerate(decoded):\n",
    "        refs = gold_df.iloc[i][\"target_text\"].split(\"|||\")\n",
    "        scores.append(compute_rouge_scores(pred, refs))\n",
    "\n",
    "    return {\"final_rouge\": float(np.mean(scores))}\n",
    "\n",
    "\n",
    "# ===============================================================\n",
    "# 6) Optuna Trial\n",
    "# ===============================================================\n",
    "def run_trial(trial, config, train_dataset, valid_dataset, valid_df):\n",
    "\n",
    "    lr = float(trial.suggest_categorical(\"learning_rate\", config[\"optuna\"][\"search_space\"][\"learning_rate\"]))\n",
    "    warm = float(trial.suggest_categorical(\"warmup_ratio\", config[\"optuna\"][\"search_space\"][\"warmup_ratio\"]))\n",
    "    epochs = int(trial.suggest_categorical(\"num_train_epochs\", config[\"optuna\"][\"search_space\"][\"num_train_epochs\"]))\n",
    "    bs = config[\"training\"][\"per_device_train_batch_size\"]\n",
    "\n",
    "    print(f\"\\n===== DEBUG: Trial {trial.number} / lr={lr}, warm={warm}, ep={epochs} =====\\n\")\n",
    "\n",
    "    tokenizer, model = create_tokenizer_and_model(config)\n",
    "    prefix = config[\"general\"][\"prefix\"]\n",
    "\n",
    "    tok_train = train_dataset.map(\n",
    "        partial(preprocess, tokenizer=tokenizer, cfg_tokenizer=config[\"tokenizer\"], prefix=prefix),\n",
    "        batched=True, remove_columns=train_dataset.column_names\n",
    "    )\n",
    "    tok_valid = valid_dataset.map(\n",
    "        partial(preprocess, tokenizer=tokenizer, cfg_tokenizer=config[\"tokenizer\"], prefix=prefix),\n",
    "        batched=True, remove_columns=valid_dataset.column_names\n",
    "    )\n",
    "\n",
    "    wandb.init(\n",
    "        project=config[\"wandb\"][\"project\"],\n",
    "        entity=config[\"wandb\"][\"entity\"],\n",
    "        name=f\"{config['wandb']['name']}_trial{trial.number}\",\n",
    "        config={\"lr\": lr, \"warmup\": warm, \"epochs\": epochs},\n",
    "        mode=config[\"wandb\"][\"mode\"],\n",
    "        reinit=True,\n",
    "    )\n",
    "\n",
    "    args = Seq2SeqTrainingArguments(\n",
    "        output_dir=f\"{config['general']['output_dir']}/trial_{trial.number}\",\n",
    "        learning_rate=lr,\n",
    "        warmup_ratio=warm,\n",
    "        num_train_epochs=epochs,\n",
    "        per_device_train_batch_size=bs,\n",
    "        gradient_accumulation_steps=config[\"training\"][\"gradient_accumulation_steps\"],\n",
    "        logging_steps=100,\n",
    "        predict_with_generate=True,\n",
    "        save_strategy=\"epoch\",\n",
    "        save_total_limit=1,\n",
    "        evaluation_strategy=\"no\",\n",
    "        fp16=config[\"training\"][\"fp16\"],\n",
    "        report_to=\"wandb\",\n",
    "    )\n",
    "\n",
    "    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=tok_train)\n",
    "\n",
    "    trainer.train()\n",
    "    preds = trainer.predict(tok_valid).predictions\n",
    "\n",
    "    score = compute_metrics((preds, None), tokenizer, valid_df)[\"final_rouge\"]\n",
    "\n",
    "    wandb.log({\"final_rouge\": score})\n",
    "    wandb.finish()\n",
    "\n",
    "    del trainer, model, tokenizer, tok_train, tok_valid\n",
    "    torch.cuda.empty_cache()\n",
    "\n",
    "    return score\n",
    "\n",
    "\n",
    "# ===============================================================\n",
    "# 7) Main — Optuna → Seed Ensemble\n",
    "# ===============================================================\n",
    "def main():\n",
    "\n",
    "    print(\"\\n===== DEBUG: Global GPU =====\")\n",
    "    if torch.cuda.is_available():\n",
    "        print(\"GPU:\", torch.cuda.get_device_name(0))\n",
    "    print(\"========================================\\n\")\n",
    "\n",
    "    prefix = config[\"general\"][\"prefix\"]\n",
    "\n",
    "    # Preview only\n",
    "    temp_tok = T5TokenizerFast.from_pretrained(config[\"general\"][\"model_name\"])\n",
    "    debug_tokenization_preview(temp_tok, config[\"tokenizer\"], prefix, train_dataset)\n",
    "    del temp_tok\n",
    "\n",
    "    # ---------------------------\n",
    "    # OPTUNA\n",
    "    # ---------------------------\n",
    "    if config[\"optuna\"][\"use\"]:\n",
    "        print(\"🔥 Optuna Search 시작!\")\n",
    "\n",
    "        study = optuna.create_study(direction=\"maximize\")\n",
    "        study.optimize(\n",
    "            lambda tr: run_trial(tr, config, train_dataset, valid_dataset, valid_df_pandas),\n",
    "            n_trials=config[\"optuna\"][\"n_trials\"],\n",
    "        )\n",
    "\n",
    "        best = study.best_trial.params\n",
    "        best_lr = float(best[\"learning_rate\"])\n",
    "        best_warm = float(best[\"warmup_ratio\"])\n",
    "        best_ep = int(best[\"num_train_epochs\"])\n",
    "\n",
    "        print(\"\\n🎉 Best Trial:\", best)\n",
    "    else:\n",
    "        best_lr = config[\"training\"][\"learning_rate\"]\n",
    "        best_warm = config[\"training\"][\"warmup_ratio\"]\n",
    "        best_ep = config[\"training\"][\"num_train_epochs\"]\n",
    "\n",
    "    # ---------------------------\n",
    "    # SEED ENSEMBLE\n",
    "    # ---------------------------\n",
    "    seeds = [42, 2025]\n",
    "    print(f\"\\n🔧 Seed Ensemble 시작 — seeds = {seeds}\")\n",
    "\n",
    "    for seed in seeds:\n",
    "        print(f\"\\n===== 🚀 Seed {seed} =====\")\n",
    "        set_seed(seed)\n",
    "\n",
    "        tokenizer, model = create_tokenizer_and_model(config)\n",
    "\n",
    "        tok_train = train_dataset.map(\n",
    "            partial(preprocess, tokenizer=tokenizer, cfg_tokenizer=config[\"tokenizer\"], prefix=prefix),\n",
    "            batched=True, remove_columns=train_dataset.column_names\n",
    "        )\n",
    "        tok_valid = valid_dataset.map(\n",
    "            partial(preprocess, tokenizer=tokenizer, cfg_tokenizer=config[\"tokenizer\"], prefix=prefix),\n",
    "            batched=True, remove_columns=valid_dataset.column_names\n",
    "        )\n",
    "\n",
    "        wandb.init(\n",
    "            project=config[\"wandb\"][\"project\"],\n",
    "            entity=config[\"wandb\"][\"entity\"],\n",
    "            name=f\"{config['wandb']['name']}_seed{seed}\",\n",
    "            config={\"lr\": best_lr, \"warmup\": best_warm, \"epochs\": best_ep, \"seed\": seed},\n",
    "            mode=config[\"wandb\"][\"mode\"],\n",
    "            reinit=True,\n",
    "        )\n",
    "\n",
    "        args = Seq2SeqTrainingArguments(\n",
    "            output_dir=f\"{config['general']['output_dir']}/seed_{seed}\",\n",
    "            learning_rate=best_lr,\n",
    "            warmup_ratio=best_warm,\n",
    "            num_train_epochs=best_ep,\n",
    "            per_device_train_batch_size=config[\"training\"][\"per_device_train_batch_size\"],\n",
    "            gradient_accumulation_steps=config[\"training\"][\"gradient_accumulation_steps\"],\n",
    "            save_strategy=\"epoch\",\n",
    "            save_total_limit=2,\n",
    "            predict_with_generate=True,\n",
    "            logging_steps=100,\n",
    "            fp16=config[\"training\"][\"fp16\"],\n",
    "            report_to=\"wandb\",\n",
    "        )\n",
    "\n",
    "        trainer = Seq2SeqTrainer(\n",
    "            model=model,\n",
    "            args=args,\n",
    "            train_dataset=tok_train,\n",
    "            eval_dataset=tok_valid,\n",
    "            tokenizer=tokenizer,\n",
    "            compute_metrics=lambda x: compute_metrics(x, tokenizer, valid_df_pandas),\n",
    "        )\n",
    "\n",
    "        trainer.train()\n",
    "        evals = trainer.evaluate()\n",
    "        print(f\"📊 Seed {seed} Eval:\", evals)\n",
    "\n",
    "        wandb.finish()\n",
    "\n",
    "        del trainer, model, tokenizer, tok_train, tok_valid\n",
    "        torch.cuda.empty_cache()\n",
    "\n",
    "    print(\"\\n🎉 전체 Seed Ensemble 완료!\")\n",
    "\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    main()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "4cfa8c48-d34e-4cb0-af81-42be0a61a70a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "📌 Test size: 499 rows\n",
      "✅ 발견된 seed 디렉토리: ['../outputs/t5_v1_large/seed_2025', '../outputs/t5_v1_large/seed_42']\n",
      "📌 [../outputs/t5_v1_large/seed_2025] best checkpoint 선택: ../outputs/t5_v1_large/seed_2025/checkpoint-5604\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/data/ephemeral/home/py310/lib/python3.10/site-packages/huggingface_hub/file_download.py:942: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.\n",
      "  warnings.warn(\n",
      "Generating: 100%|██████████████████| 250/250 [06:33<00:00,  1.57s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "📁 Saved → submission_seed_2025.csv\n",
      "📌 [../outputs/t5_v1_large/seed_42] best checkpoint 선택: ../outputs/t5_v1_large/seed_42/checkpoint-5604\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Generating: 100%|██████████████████| 250/250 [06:28<00:00,  1.55s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "📁 Saved → submission_seed_42.csv\n",
      "📁 Saved → submission_ensemble.csv\n"
     ]
    }
   ],
   "source": [
    "# ===============================================================\n",
    "# ensemble_inference.py — Seed별 T5 요약 + Ensemble + 제출 파일 생성 (fname 포함)\n",
    "#  - train.py의 output_dir/seed_* 구조 자동 탐색\n",
    "#  - 각 seed별 inference + ensemble inference 생성\n",
    "#  - 제출 형식: fname, summary 두 컬럼\n",
    "# ===============================================================\n",
    "\n",
    "import os\n",
    "import yaml\n",
    "import torch\n",
    "import pandas as pd\n",
    "from tqdm import tqdm\n",
    "from transformers import T5ForConditionalGeneration, T5TokenizerFast\n",
    "\n",
    "torch.backends.cuda.matmul.allow_tf32 = True\n",
    "DEVICE = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
    "\n",
    "\n",
    "# ===============================================================\n",
    "# 1) Load config\n",
    "# ===============================================================\n",
    "def load_config(path=\"config.yaml\"):\n",
    "    with open(path, \"r\") as f:\n",
    "        return yaml.safe_load(f)\n",
    "\n",
    "config = load_config()\n",
    "\n",
    "\n",
    "# ===============================================================\n",
    "# 2) seed_* 디렉토리 탐색\n",
    "# ===============================================================\n",
    "def find_seed_dirs(base_output_dir):\n",
    "    if not os.path.exists(base_output_dir):\n",
    "        raise FileNotFoundError(f\"❌ output_dir 없음: {base_output_dir}\")\n",
    "\n",
    "    seed_dirs = [\n",
    "        os.path.join(base_output_dir, d)\n",
    "        for d in os.listdir(base_output_dir)\n",
    "        if d.startswith(\"seed_\") and os.path.isdir(os.path.join(base_output_dir, d))\n",
    "    ]\n",
    "\n",
    "    if not seed_dirs:\n",
    "        print(f\"⚠️ seed_* 없음 → 단일 모델로 처리: {base_output_dir}\")\n",
    "        return [base_output_dir]\n",
    "\n",
    "    print(f\"✅ 발견된 seed 디렉토리: {seed_dirs}\")\n",
    "    return seed_dirs\n",
    "\n",
    "\n",
    "def get_best_checkpoint_path_in_dir(model_dir):\n",
    "    subdirs = [\n",
    "        d for d in os.listdir(model_dir)\n",
    "        if d.startswith(\"checkpoint\") and os.path.isdir(os.path.join(model_dir, d))\n",
    "    ]\n",
    "\n",
    "    if not subdirs:\n",
    "        print(f\"📌 [{model_dir}] checkpoint-* 없음 → 본 디렉토리 사용\")\n",
    "        return model_dir\n",
    "\n",
    "    subdirs = sorted(subdirs, key=lambda x: int(x.split(\"-\")[-1]))\n",
    "    best = os.path.join(model_dir, subdirs[-1])\n",
    "    print(f\"📌 [{model_dir}] best checkpoint 선택: {best}\")\n",
    "    return best\n",
    "\n",
    "\n",
    "# ===============================================================\n",
    "# 3) model + tokenizer 로드\n",
    "# ===============================================================\n",
    "def load_model_and_tokenizer_for_checkpoint(cfg, checkpoint_path):\n",
    "    tokenizer = T5TokenizerFast.from_pretrained(cfg[\"general\"][\"model_name\"])\n",
    "    tokenizer.add_tokens(cfg[\"tokenizer\"][\"special_tokens\"])\n",
    "\n",
    "    model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)\n",
    "    model.resize_token_embeddings(len(tokenizer))\n",
    "\n",
    "    model.to(DEVICE)\n",
    "    model.eval()\n",
    "\n",
    "    return tokenizer, model\n",
    "\n",
    "\n",
    "# ===============================================================\n",
    "# 4) Test dataset 로드 (fname 유지!)\n",
    "# ===============================================================\n",
    "def load_test_dataset(cfg):\n",
    "    test_path = os.path.join(cfg[\"general\"][\"data_dir\"], cfg[\"general\"][\"test_file\"])\n",
    "    df = pd.read_csv(test_path)\n",
    "\n",
    "    if \"dialogue_clean\" not in df.columns:\n",
    "        raise ValueError(\"❌ test.csv에 dialogue_clean 없음\")\n",
    "\n",
    "    df = df.rename(columns={\"dialogue_clean\": \"input_text\"})\n",
    "    print(f\"📌 Test size: {len(df)} rows\")\n",
    "    return df  # fname 포함됨\n",
    "\n",
    "\n",
    "# ===============================================================\n",
    "# 5) 요약 생성 + 특수토큰 제거\n",
    "# ===============================================================\n",
    "def clean_summary(text, remove_tokens):\n",
    "    for t in remove_tokens:\n",
    "        text = text.replace(t, \"\")\n",
    "    return text.strip()\n",
    "\n",
    "\n",
    "def generate_summaries_for_model(model, tokenizer, df, cfg):\n",
    "    results = []\n",
    "\n",
    "    batch_size = cfg[\"inference\"][\"batch_size\"]\n",
    "    num_beams = cfg[\"inference\"][\"num_beams\"]\n",
    "    max_len = cfg[\"inference\"][\"max_length\"]\n",
    "    no_repeat = cfg[\"inference\"][\"no_repeat_ngram_size\"]\n",
    "    prefix = cfg[\"general\"].get(\"prefix\", \"\")\n",
    "    remove_tokens = cfg[\"inference\"][\"remove_tokens\"]\n",
    "\n",
    "    for i in tqdm(range(0, len(df), batch_size), desc=\"Generating\"):\n",
    "        batch_text = [prefix + x for x in df[\"input_text\"].iloc[i:i+batch_size].tolist()]\n",
    "\n",
    "        inputs = tokenizer(\n",
    "            batch_text,\n",
    "            max_length=cfg[\"tokenizer\"][\"encoder_max_len\"],\n",
    "            truncation=True,\n",
    "            padding=True,\n",
    "            return_tensors=\"pt\"\n",
    "        ).to(DEVICE)\n",
    "\n",
    "        with torch.no_grad():\n",
    "            outputs = model.generate(\n",
    "                **inputs,\n",
    "                max_length=max_len,\n",
    "                num_beams=num_beams,\n",
    "                no_repeat_ngram_size=no_repeat,\n",
    "                early_stopping=True,\n",
    "            )\n",
    "\n",
    "        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)\n",
    "        cleaned = [clean_summary(t, remove_tokens) for t in decoded]\n",
    "        results.extend(cleaned)\n",
    "\n",
    "    return results\n",
    "\n",
    "\n",
    "# ===============================================================\n",
    "# 6) Ensemble (중간 길이에 가장 근접한 summary)\n",
    "# ===============================================================\n",
    "def ensemble_summaries(seed_to_summaries):\n",
    "    seeds = list(seed_to_summaries.keys())\n",
    "    N = len(seed_to_summaries[seeds[0]])\n",
    "\n",
    "    final = []\n",
    "    for i in range(N):\n",
    "        cands = [seed_to_summaries[s][i] for s in seeds]\n",
    "        lens = [len(c) for c in cands]\n",
    "        avg_len = sum(lens) / len(lens)\n",
    "\n",
    "        best = min(range(len(cands)), key=lambda j: abs(len(cands[j]) - avg_len))\n",
    "        final.append(cands[best])\n",
    "\n",
    "    return final\n",
    "\n",
    "\n",
    "# ===============================================================\n",
    "# 7) 제출 저장 (fname + summary)\n",
    "# ===============================================================\n",
    "def save_submission(fname_list, summaries, path):\n",
    "    df = pd.DataFrame({\"fname\": fname_list, \"summary\": summaries})\n",
    "    df.to_csv(path, index=False)\n",
    "    print(f\"📁 Saved → {path}\")\n",
    "\n",
    "\n",
    "# ===============================================================\n",
    "# Main\n",
    "# ===============================================================\n",
    "def main():\n",
    "    base_dir = config[\"general\"][\"output_dir\"]\n",
    "\n",
    "    # test load\n",
    "    test_df = load_test_dataset(config)\n",
    "    fnames = test_df[\"fname\"].tolist()\n",
    "\n",
    "    # seed dirs\n",
    "    seed_dirs = find_seed_dirs(base_dir)\n",
    "    seed_labels = [os.path.basename(x) for x in seed_dirs]\n",
    "\n",
    "    seed_to_summaries = {}\n",
    "\n",
    "    # 각 seed별 inference 실행\n",
    "    for seed_dir, seed_label in zip(seed_dirs, seed_labels):\n",
    "        ckpt_path = get_best_checkpoint_path_in_dir(seed_dir)\n",
    "\n",
    "        tokenizer, model = load_model_and_tokenizer_for_checkpoint(config, ckpt_path)\n",
    "        summaries = generate_summaries_for_model(model, tokenizer, test_df, config)\n",
    "\n",
    "        seed_to_summaries[seed_label] = summaries\n",
    "\n",
    "        save_submission(\n",
    "            fnames,\n",
    "            summaries,\n",
    "            f\"submission_{seed_label}.csv\"\n",
    "        )\n",
    "\n",
    "        del model\n",
    "        del tokenizer\n",
    "        torch.cuda.empty_cache()\n",
    "\n",
    "    # 앙상블\n",
    "    if len(seed_to_summaries) > 1:\n",
    "        ensemble = ensemble_summaries(seed_to_summaries)\n",
    "        save_submission(fnames, ensemble, \"submission_ensemble.csv\")\n",
    "    else:\n",
    "        only = seed_labels[0]\n",
    "        save_submission(fnames, seed_to_summaries[only], \"submission.csv\")\n",
    "\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    main()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "511f9f40-623a-4026-aa92-4295a6e77b7e",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python (py310)",
   "language": "python",
   "name": "py310"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
