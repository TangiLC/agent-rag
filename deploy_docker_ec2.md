# Déploiement Docker sur EC2 (build local + push ECR + Nginx)

Ce guide décrit un déploiement simple et reproductible : build local, push ECR, pull sur EC2, reverse proxy Nginx, HTTPS via Let’s Encrypt.

---

## 1) Pré-requis

- Un compte AWS avec accès ECR/EC2.
- Un nom de domaine pointant vers l’IP publique EC2.
- AWS CLI installé et configuré en local (`aws configure`).
- Docker installé en local.

---

## 2) Créer le dépôt ECR

```bash
aws ecr create-repository --repository-name tangi-rag-agent --region eu-west-1
```

URI du dépôt (déjà créé) :

`947278783763.dkr.ecr.eu-west-1.amazonaws.com/tangi-rag-agent`

---

## 3) Build local + push vers ECR

```bash
# Build
docker build -t agent-rag-streamlit:latest .

# Login ECR
aws ecr get-login-password --region eu-west-1 | \
  docker login --username AWS --password-stdin 947278783763.dkr.ecr.eu-west-1.amazonaws.com

# Tag + push
docker tag agent-rag-streamlit:latest 947278783763.dkr.ecr.eu-west-1.amazonaws.com/tangi-rag-agent:latest
docker push 947278783763.dkr.ecr.eu-west-1.amazonaws.com/tangi-rag-agent:latest
```

---

## 4) Provisionner l’instance EC2

Depuis la console AWS, lance une instance avec ces paramètres :

- AMI : `Ubuntu Server 22.04 LTS (x86_64)`.
- Type : `t3.large` minimum recommandé (CPU build + inference locale), `t3.xlarge` plus confortable.
- Stockage : `30 Go` minimum, `60 Go` recommandé si modèles/corpus volumineux.
- Key pair : crée ou sélectionne une clé `.pem` pour SSH.
- Public IP : activée.
- Security Group inbound :
- `22` (SSH) : idéalement limité à ton IP publique.
- `80` (HTTP) : `0.0.0.0/0`.
- `443` (HTTPS) : `0.0.0.0/0`.

Associe un IAM Role à l’instance pour accéder à ECR sans clés statiques :

- Policy minimale : `AmazonEC2ContainerRegistryReadOnly`.
- Attache la role à l’instance (EC2 > Actions > Security > Modify IAM role).

Vérifie ensuite que ton DNS pointe vers l’IP publique EC2 :

- Enregistrement `A` : `<your-domain>` -> `<EC2_PUBLIC_IP>`.
- Si propagation DNS en cours, attends avant `certbot`.

Connecte-toi en SSH :

```bash
ssh -i /path/to/key.pem ubuntu@<EC2_PUBLIC_IP>
```

---

## 5) Installer Docker et Nginx sur EC2

```bash
sudo apt update
sudo apt install -y docker.io nginx
sudo usermod -aG docker $USER
newgrp docker
```

---

## 6) Préparer les volumes et données

```bash
sudo mkdir -p /srv/agent-rag/models /srv/agent-rag/rag_docs /srv/agent-rag/data
```

- Mets ton modèle GGUF dans `/srv/agent-rag/models/`
- Mets ton corpus PDF dans `/srv/agent-rag/rag_docs/`
- Mets les artefacts FAISS/embeddings dans `/srv/agent-rag/data/`

---

## 7) Pull de l’image depuis ECR

```bash
aws ecr get-login-password --region eu-west-1 | \
  docker login --username AWS --password-stdin 947278783763.dkr.ecr.eu-west-1.amazonaws.com

docker pull 947278783763.dkr.ecr.eu-west-1.amazonaws.com/tangi-rag-agent:latest
```

---

## 8) Lancer le conteneur

```bash
docker run -d --name rag \
  -p 127.0.0.1:8501:8501 \
  -v /srv/agent-rag/models:/app/models \
  -v /srv/agent-rag/rag_docs:/app/rag_docs \
  -v /srv/agent-rag/data:/app/data \
  --restart unless-stopped \
  947278783763.dkr.ecr.eu-west-1.amazonaws.com/tangi-rag-agent:latest
```

---

## 9) Configurer Nginx (reverse proxy)

Créer le vhost :

```bash
sudo tee /etc/nginx/sites-available/agent-rag > /dev/null <<'NGINX'
server {
    listen 80;
    server_name <your-domain>;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
NGINX
```

Activer et reload :

```bash
sudo ln -s /etc/nginx/sites-available/agent-rag /etc/nginx/sites-enabled/agent-rag
sudo nginx -t
sudo systemctl reload nginx
```

---

## 10) Activer HTTPS (Let’s Encrypt)

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d <your-domain>
```

---

## 11) Vérifications

- UI: `https://<your-domain>`
- Logs conteneur:

```bash
docker logs -f rag
```

---

## 12) Mises à jour

Quand tu pushes une nouvelle image :

```bash
docker pull 947278783763.dkr.ecr.eu-west-1.amazonaws.com/tangi-rag-agent:latest
docker stop rag && docker rm rag
# relancer le docker run (section 8)
```

---

## Notes pratiques

- Si tu n’as pas de modèle GGUF, définis `USE_LLM=false` en variable d’environnement dans le `docker run`.
- Le port Streamlit n’est pas exposé publiquement : Nginx fait le proxy vers `127.0.0.1:8501`.
