function highlighted = create_damage_highlight(img, damageMask)
%CREATE_DAMAGE_HIGHLIGHT Overlay detected damage pixels in red.

highlighted = img;
R = highlighted(:, :, 1);
G = highlighted(:, :, 2);
B = highlighted(:, :, 3);
R(damageMask) = 255;
G(damageMask) = 0;
B(damageMask) = 0;
highlighted(:, :, 1) = R;
highlighted(:, :, 2) = G;
highlighted(:, :, 3) = B;
end
