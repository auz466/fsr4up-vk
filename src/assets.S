/* AMD's shaders and initializer data, as tools/setup.py packed them (build/gen/assets.bin) */
    .section .rodata
    .global fsr4up_assets
    .hidden fsr4up_assets
    .type fsr4up_assets, @object
    .balign 16
fsr4up_assets:
    .incbin "assets.bin"
    .size fsr4up_assets, . - fsr4up_assets
    .section .note.GNU-stack,"",@progbits
