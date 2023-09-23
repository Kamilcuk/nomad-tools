Vagrant.configure("2") do |config|
  config.vm.box = "bento/ubuntu-22.04"
  config.vm.hostname = "nomad-tools"
  config.vm.synced_folder ".", "/app"
  config.vm.define "nomad-tools"
  config.vm.provision :docker
  config.vm.provision :shell, path: "./tests/provision.sh", args: "vagrant"
  config.vm.network "forwarded_port", guest: 4646, host: 64646
end
