tls {
  http                   = true
  rpc                    = true
  ca_file                = "./tests/nomad.d/nomad-agent-ca.pem"
  cert_file              = "./tests/nomad.d/global-server-nomad.pem"
  key_file               = "./tests/nomad.d/global-server-nomad-key.pem"
  verify_server_hostname = true
  verify_https_client    = true
}
