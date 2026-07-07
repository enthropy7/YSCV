{ inputs, ... }: {
  perSystem =
    {
      pkgs,
      lib,
      self',
      system,
      ...
    }:
    let
      isLinux = pkgs.stdenv.isLinux;
      isDarwin = pkgs.stdenv.isDarwin;
      isAarch64 = pkgs.stdenv.hostPlatform.isAarch64;
      isX86_64 = pkgs.stdenv.hostPlatform.isx86_64;

      craneLib = (inputs.crane.mkLib pkgs).overrideToolchain (
        p:
        p.rust-bin.stable.latest.default.override {
          extensions = [
            "rust-analyzer"
            "rust-src"
            "rustfmt"
            "clippy"
          ];
        }
      );

      commonArgs = {
        inherit (craneLib.crateNameFromCargoToml { cargoToml = ./Cargo.toml; }) version;

        # This should be filtered, but there are a lot of useful extensions for tests
        src = ./.;
        strictDeps = true;

        nativeBuildInputs =
          with pkgs;
          [
            pkg-config
            protobuf
            rustPlatform.bindgenHook
          ]
          ++ lib.optionals isLinux [
            linuxHeaders
          ];

        buildInputs = lib.optionals isLinux [
          pkgs.openblas
          # TODO : mkl, armpl
        ];

        env = {
          PROTOC = "${pkgs.protobuf}/bin/protoc";
          OPENBLAS_DIR = lib.optionalString isLinux "${pkgs.openblas}";
        };
      };
    in
    {
      _module.args.pkgs = import inputs.nixpkgs {
        inherit system;

        overlays = [
          inputs.rust-overlay.overlays.default
        ];
      };

      checks =
        let
          testFeatures = [
            "blas"
            "gpu"
            "native-camera"
          ]
          ++ lib.optionals isDarwin [
            "metal-backend"
          ]
          ++ lib.optionals (isLinux && isAarch64) [
            "rknn"
            # "armpl"
          ]
          ++ lib.optionals (isLinux && isX86_64) [
            # "mkl"
          ];

          mkTest = feature: {
            name = "test-${feature}";
            value = craneLib.cargoTest (
              commonArgs
              // {
                pname = "yscv-${feature}";
                cargoExtraArgs = "--workspace --features ${feature}";

                cargoArtifacts = null;
              }
            );
          };
        in
        builtins.listToAttrs (map mkTest testFeatures);

      devShells.default = craneLib.devShell {
        inherit (commonArgs) env;

        packages = commonArgs.buildInputs ++ commonArgs.nativeBuildInputs;
      };
    };
}
