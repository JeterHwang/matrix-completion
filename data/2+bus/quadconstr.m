function [Ap, Aq, Av, Apf, Aqf, Apt, Aqt] = quadconstr(Ybus, Yf, f, Yt, t)
%PFCONSTR Generate linear constraints for power flow
% Let v = [vr; vi], 
% Then Ap' * kron(v,v) = real power
%      Aq' * kron(v,v) = imag power
%      Av' * kron(v,v) = voltage mag squared
%      Apf'* kron(v,v) = real power sent
%      Aqf'* kron(v,v) = imag power sent
%      Apt'* kron(v,v) = imag power received
%      Aqt'* kron(v,v) = imag power received

nbus = length(Ybus);
Id = speye(nbus);

% Bus constraints
[Ap, Aq, Av] = deal(cell(1,nbus));
for k = 1:nbus
    % P.94 of Lavaei & Low 2012
    Ek = Id(:,k) * Id(:,k)';
    Ysum = Ek*Ybus + Ybus.'*Ek;
    Ydif = Ek*Ybus - Ybus.'*Ek;
    Ap{k} =  0.5 *[real(Ysum), -imag(Ydif);
                   imag(Ydif),  real(Ysum)];
    Aq{k} = -0.5 *[imag(Ysum), real(Ydif);
                  -real(Ydif), imag(Ysum)];
    Av{k} = blkdiag(Ek,Ek);
    
    % Vectorize
    Ap{k} = Ap{k}(:);
    Aq{k} = Aq{k}(:);
    Av{k} = Av{k}(:);
end
Ap = [Ap{:}];
Aq = [Aq{:}];
Av = [Av{:}];

% Line constraints (from)
if nargin > 1
    nlines = size(Yf,1); Yf = Yf';
    [Apf, Aqf] = deal(cell(1,nlines));
    for k = 1:nlines
        % P.94 of Lavaei & Low 2012
        Ylm = Yf(:,k)*Id(:,f(k))';
        Ysum = Ylm + Ylm.';
        Ydif = Ylm - Ylm.';
        Apf{k} =  0.5 *[real(Ysum), -imag(Ydif);
                       imag(Ydif),  real(Ysum)];
        Aqf{k} =  0.5 *[imag(Ysum), real(Ydif);
                      -real(Ydif), imag(Ysum)];

        % Vectorize
        Apf{k} = Apf{k}(:);
        Aqf{k} = Aqf{k}(:);
    end
    Apf = [Apf{:}];
    Aqf = [Aqf{:}];
end

% Line constraints (to)
if nargin > 1
    nlines = size(Yt,1); Yt = Yt';
    [Apt, Aqt] = deal(cell(1,nlines));
    for k = 1:nlines
        % P.94 of Lavaei & Low 2012
        Ylm = Yt(:,k)*Id(:,t(k))';
        Ysum = Ylm + Ylm.';
        Ydif = Ylm - Ylm.';
        Apt{k} =  0.5 *[real(Ysum), -imag(Ydif);
                       imag(Ydif),  real(Ysum)];
        Aqt{k} =  0.5 *[imag(Ysum), real(Ydif);
                      -real(Ydif), imag(Ysum)];

        % Vectorize
        Apt{k} = Apt{k}(:);
        Aqt{k} = Aqt{k}(:);
    end
    Apt = [Apt{:}];
    Aqt = [Aqt{:}];
end
end