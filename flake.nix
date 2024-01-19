{
    inputs = {
        nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.05";
        flake-utils = {
            url = "github:numtide/flake-utils";
            inputs.nixpkgs.follows = "nixpkgs";
        };
        rust-overlay = {
            url = "github:oxalica/rust-overlay";
            inputs.nixpkgs.follows = "nixpkgs";
        };
    };
    outputs = { self, nixpkgs, flake-utils, rust-overlay }:
        let
            system = "x86_64-linux";
            overlays = [ (import rust-overlay) ];
            pkgs = import nixpkgs {
                inherit system overlays;
                config.allowUnfree = true;
            };
            nativeBuildInputs = with pkgs; [ 
                rust-bin.stable.latest.default
                pkg-config
                glibc.dev
                cmake
                fontconfig

                xorg.libX11
                xorg.libXcursor
                xorg.libXrandr
                xorg.libXi

                vulkan-headers
                vulkan-validation-layers

                rust-analyzer
            ];

            dlopenLibs = with pkgs; [
                vulkan-loader
                libxkbcommon
            ];
            shellHook = "
              export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${pkgs.lib.makeLibraryPath dlopenLibs}
              export PATH=$PATH:$HOME/.cargo/bin
            ";
        in {
            devShells.${system}.default = pkgs.mkShell {
                inherit nativeBuildInputs shellHook;
                buildInputs = [  ];
                # VULKAN_SDK = "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d";
            };
        };
}

